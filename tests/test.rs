use rivulus::{
    build_indices, eval_predicate_mask, filter_batch, filter_with_indices, filter_with_mask,
    project_batch, Column, ColumnView, EvalError, Expr, FilterScratch, FilterStrategy, Pipeline,
    ProjItem, ProjectScratch, Scan, Table,
};

/* ---------- generators & helper test ---------- */

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn build_table(n: usize, seed: u64) -> Table {
    let mut s1 = seed ^ 0x9E3779B97F4A7C15;
    let mut s2 = seed.wrapping_add(0xBF58476D1CE4E5B9);
    const M: i64 = 1_000_000;

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut mask = Vec::with_capacity(n);

    for _ in 0..n {
        let xi = (lcg(&mut s1) % (M as u64)) as i64;
        let yi = (lcg(&mut s2) % (M as u64)) as i64;
        x.push(xi);
        y.push(yi);
        mask.push(xi % 10 == 0); // ~10% true
    }

    Table {
        schema: ["x", "y", "mask_10pct"],
        cols: [Column::I64(x), Column::I64(y), Column::Bool(mask)],
        len: n,
    }
}

fn concat_from_scan(table: &Table, chunk: usize) -> (Vec<usize>, Vec<i64>, Vec<i64>, Vec<bool>) {
    let mut scan = Scan::with_chunk(table, chunk);
    let mut lens = Vec::new();
    let mut xs = Vec::with_capacity(table.len);
    let mut ys = Vec::with_capacity(table.len);
    let mut ms = Vec::with_capacity(table.len);

    while let Some(batch) = scan.next_batch() {
        lens.push(batch.len);
        match &batch.cols[0] {
            ColumnView::I64(s) => xs.extend_from_slice(s),
            _ => panic!(),
        }
        match &batch.cols[1] {
            ColumnView::I64(s) => ys.extend_from_slice(s),
            _ => panic!(),
        }
        match &batch.cols[2] {
            ColumnView::Bool(s) => ms.extend_from_slice(s),
            _ => panic!(),
        }
        debug_assert!(batch.cols.iter().all(|c| c.len() == batch.len));
    }
    (lens, xs, ys, ms)
}

/* ---------- TEST: Milestone 1 (Scan) ---------- */

#[test]
fn scan_batch_lengths_and_reconstruct_small() {
    let t = build_table(10, 42);
    let (lens, xs, ys, ms) = concat_from_scan(&t, 4);

    // lunghezze attese: [4,4,2]
    assert_eq!(lens, vec![4, 4, 2]);

    // ricostruzione colonne
    let (x0, y0, m0) = match (&t.cols[0], &t.cols[1], &t.cols[2]) {
        (Column::I64(a), Column::I64(b), Column::Bool(c)) => (a, b, c),
        _ => panic!(),
    };
    assert_eq!(&xs, x0);
    assert_eq!(&ys, y0);
    assert_eq!(&ms, m0);
}

#[test]
fn scan_edges_exact_and_plus_one() {
    let cases: &[(usize, usize, &[usize])] = &[
        (8192, 8192, &[8192]),
        (8193, 8192, &[8192, 1]),
        (0, 8192, &[]),
    ];

    for &(n, chunk, expected) in cases {
        let t = build_table(n, 7);
        let (lens, ..) = concat_from_scan(&t, chunk);
        assert_eq!(lens.as_slice(), expected, "N={n}, chunk={chunk}");
    }
}

/* ---------- TEST: Milestone 2 (Expr, unit) ---------- */

#[test]
fn expr_lit_i64_broadcasts_to_batch_len() {
    let t = build_table(10, 1);
    let mut scan = Scan::with_chunk(&t, 6);
    let batch = scan.next_batch().unwrap(); // len = 6
    let e = Expr::LitI64(7);
    let mut out = Vec::new();
    e.eval_i64(&batch, &mut out).unwrap();
    assert_eq!(out.len(), batch.len);
    assert!(out.iter().all(|&v| v == 7));
}

#[test]
fn expr_col_and_errors() {
    let t = build_table(8, 2);
    let mut scan = Scan::with_chunk(&t, 5);
    let batch = scan.next_batch().unwrap(); // len = 5

    // Col("x") come i64 = copia
    let mut out_i = Vec::new();
    Expr::Col("x").eval_i64(&batch, &mut out_i).unwrap();
    match &batch.cols[0] {
        ColumnView::I64(s) => assert_eq!(&out_i[..], &s[..]),
        _ => panic!(),
    }

    // Col("mask_10pct") in eval_bool = copia
    let mut out_b = Vec::new();
    Expr::Col("mask_10pct")
        .eval_bool(&batch, &mut out_b)
        .unwrap();
    match &batch.cols[2] {
        ColumnView::Bool(s) => assert_eq!(&out_b[..], &s[..]),
        _ => panic!(),
    }

    // Errori
    assert!(matches!(
        Expr::Col("mask_10pct").eval_i64(&batch, &mut out_i),
        Err(EvalError::TypeMismatch(_))
    ));
    assert!(matches!(
        Expr::Col("does_not_exist").eval_i64(&batch, &mut out_i),
        Err(EvalError::UnknownColumn(_))
    ));
    assert!(matches!(
        Expr::LitI64(1).eval_bool(&batch, &mut out_b),
        Err(EvalError::TypeMismatch(_))
    ));
}

#[test]
fn expr_add_and_gt_on_one_batch() {
    let t = build_table(12, 3);
    let mut scan = Scan::with_chunk(&t, 10);
    let batch = scan.next_batch().unwrap(); // len = 10

    // Add(x, 1)
    let mut add_out = Vec::new();
    Expr::Add(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(1)))
        .eval_i64(&batch, &mut add_out)
        .unwrap();
    // verità terra
    let x_slice = match &batch.cols[0] {
        ColumnView::I64(s) => s,
        _ => panic!(),
    };
    assert!(
        add_out
            .iter()
            .zip(x_slice.iter())
            .all(|(o, x)| *o == *x + 1)
    );

    // Gt(x, T)
    let mut gt_out = Vec::new();
    let tval = 500_000i64;
    Expr::Gt(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(tval)))
        .eval_bool(&batch, &mut gt_out)
        .unwrap();
    assert_eq!(gt_out.len(), batch.len);
    assert!(
        gt_out
            .iter()
            .zip(x_slice.iter())
            .all(|(b, x)| *b == (*x > tval))
    );
}

/* ---------- TEST: Integrazione (Scan + Expr su tutti i batch) ---------- */

#[test]
fn integration_scan_plus_expr_concat_matches_truth() {
    let n = 50_000;
    let chunk = 4096;
    let t = build_table(n, 12345);

    // verità terra su T intero
    let (x_full, _y_full, _) = match (&t.cols[0], &t.cols[1], &t.cols[2]) {
        (Column::I64(x), Column::I64(y), Column::Bool(m)) => (x, y, m),
        _ => panic!(),
    };
    let tval = 500_000i64;
    let truth_gt: Vec<bool> = x_full.iter().map(|&x| x > tval).collect();
    let truth_add: Vec<i64> = x_full.iter().map(|&x| x + 1).collect();

    // esecuzione batch-per-batch
    let mut scan = Scan::with_chunk(&t, chunk);
    let mut got_gt = Vec::with_capacity(n);
    let mut got_add = Vec::with_capacity(n);

    while let Some(batch) = scan.next_batch() {
        let mut mask = Vec::new();
        Expr::Gt(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(tval)))
            .eval_bool(&batch, &mut mask)
            .unwrap();
        got_gt.extend_from_slice(&mask);

        let mut add = Vec::new();
        Expr::Add(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(1)))
            .eval_i64(&batch, &mut add)
            .unwrap();
        got_add.extend_from_slice(&add);
    }

    assert_eq!(got_gt, truth_gt);
    assert_eq!(got_add, truth_add);
}

/* ---------- TEST: Milestone 3 ---------- */

#[test]
fn filter_mask_vs_indices_equivalence_toy() {
    // Batch piccolo e mask artificiale: seleziona righe con indice dispari
    let t = build_table(8, 123);
    let mut scan = Scan::with_chunk(&t, 8);
    let batch = scan.next_batch().unwrap();
    let n = batch.len;

    let mut mask = vec![false; n];
    for i in 0..n {
        mask[i] = i % 2 == 1;
    }
    let mut idx = Vec::new();
    build_indices(&mask, &mut idx);

    let out_m = filter_with_mask(&batch, &mask);
    let out_i = filter_with_indices(&batch, &idx);

    // stesse dimensioni e schema
    assert_eq!(out_m.len, out_i.len);
    assert_eq!(out_m.schema, out_i.schema);
    assert_eq!(out_m.schema.len(), out_m.cols.len());
    assert_eq!(out_i.schema.len(), out_i.cols.len());

    // colonne identiche element-wise
    assert_eq!(out_m.cols.len(), out_i.cols.len());
    for j in 0..out_m.cols.len() {
        match (&out_m.cols[j], &out_i.cols[j]) {
            (Column::I64(a), Column::I64(b)) => assert_eq!(a, b),
            (Column::Bool(a), Column::Bool(b)) => assert_eq!(a, b),
            _ => panic!("tipi colonna non corrispondenti"),
        }
    }
}

#[test]
fn filter_edges_all_false_all_true() {
    let t = build_table(16, 7);
    let mut scan = Scan::with_chunk(&t, 10);
    let batch = scan.next_batch().unwrap();
    let n = batch.len;

    // all-false
    let mask_false = vec![false; n];
    let out_f = filter_with_mask(&batch, &mask_false);
    assert_eq!(out_f.len, 0);
    assert!(out_f.cols.iter().all(|c| match c {
        Column::I64(v) => v.is_empty(),
        Column::Bool(v) => v.is_empty(),
    }));

    // all-true
    let mask_true = vec![true; n];
    let out_t = filter_with_mask(&batch, &mask_true);
    assert_eq!(out_t.len, n);

    // confronta con gli slice di input
    for (j, col) in batch.cols.iter().enumerate() {
        match (col, &out_t.cols[j]) {
            (ColumnView::I64(src), Column::I64(dst)) => assert_eq!(src, &dst.as_slice()),
            (ColumnView::Bool(src), Column::Bool(dst)) => assert_eq!(src, &dst.as_slice()),
            _ => panic!("tipo colonna non corrispondente"),
        }
    }
}

#[test]
fn filter_batch_equivalence_mask_vs_indices_varie_selettivita() {
    let t = build_table(50_000, 12345);
    let mut scan = Scan::with_chunk(&t, 8192);
    let batch = scan.next_batch().unwrap(); // test su un batch “grande”

    // Soglie per ~0%, ~5%, ~50%, ~95%, ~100% (uniforme in [0, 1_000_000))
    let thresholds = [
        1_000_000i64, // ~0% (x > 1_000_000)
        950_000i64,   // ~5%
        500_000i64,   // ~50%
        50_000i64,    // ~95%
        -1i64,        // ~100% (x > -1)
    ];

    for &tval in &thresholds {
        let pred = Expr::Gt(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(tval)));

        let mut scratch = FilterScratch {
            mask: Vec::new(),
            idx: Vec::new(),
            tmp_i64_a: Vec::new(),
            tmp_i64_b: Vec::new(),
        };

        let out_m = filter_batch(&batch, &pred, FilterStrategy::Mask, &mut scratch).unwrap();
        // costruisci gli indici poi esegui Indices
        build_indices(&scratch.mask, &mut scratch.idx);
        let out_i = filter_with_indices(&batch, &scratch.idx);

        // stesse dimensioni/schema
        assert_eq!(out_m.len, out_i.len, "soglia={}", tval);
        assert_eq!(out_m.schema, out_i.schema);

        // colonne identiche
        for j in 0..out_m.cols.len() {
            match (&out_m.cols[j], &out_i.cols[j]) {
                (Column::I64(a), Column::I64(b)) => assert_eq!(a, b, "col {} (I64), t={}", j, tval),
                (Column::Bool(a), Column::Bool(b)) => {
                    assert_eq!(a, b, "col {} (Bool), t={}", j, tval)
                }
                _ => panic!("tipo colonna non corrispondente"),
            }
        }
    }
}

#[test]
fn eval_predicate_mask_len_e_build_indices_count() {
    let t = build_table(10_000, 77);
    let mut scan = Scan::with_chunk(&t, 4096);
    let batch = scan.next_batch().unwrap();
    let pred = Expr::Gt(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(700_000)));

    let mut mask = Vec::new();
    eval_predicate_mask(&pred, &batch, &mut mask).unwrap();
    assert_eq!(mask.len(), batch.len);

    let true_count = mask.iter().filter(|&&b| b).count();
    let mut idx = Vec::new();
    build_indices(&mask, &mut idx);
    assert_eq!(idx.len(), true_count);
    // indici in ordine crescente
    assert!(idx.windows(2).all(|w| w[0] < w[1]));
    // tutti validi
    assert!(idx.iter().all(|&i| i < batch.len));
}

#[test]
fn filter_batch_errori_predicato() {
    let t = build_table(128, 5);
    let mut scan = Scan::with_chunk(&t, 64);
    let batch = scan.next_batch().unwrap();

    // Predicato non booleano: Add -> TypeMismatch
    let pred_bad = Expr::Add(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(1)));
    let mut scratch = FilterScratch {
        mask: Vec::new(),
        idx: Vec::new(),
        tmp_i64_a: Vec::new(),
        tmp_i64_b: Vec::new(),
    };

    let err = filter_batch(&batch, &pred_bad, FilterStrategy::Mask, &mut scratch);
    assert!(matches!(err, Err(EvalError::TypeMismatch(_))));

    let pred_unk = Expr::Col("does_not_exist"); // eval_bool fallirà con UnknownColumn
    let err2 = filter_batch(&batch, &pred_unk, FilterStrategy::Mask, &mut scratch);
    assert!(matches!(err2, Err(EvalError::UnknownColumn(_))));
}

/* ---------- TEST: Milestone 4 ---------- */

#[test]
fn project_keep_only_identity() {
    // N=12, prendiamo un batch non pieno per testare che usiamo batch.len
    let t = build_table(12, 42);
    let mut scan = Scan::with_chunk(&t, 7);
    let batch = scan.next_batch().unwrap();

    // Keep: stessi nomi, stesso ordine
    let items = &[
        ProjItem::Keep {
            name: "x",
            alias: None,
        },
        ProjItem::Keep {
            name: "y",
            alias: None,
        },
        ProjItem::Keep {
            name: "mask_10pct",
            alias: None,
        },
    ];

    let mut scratch = ProjectScratch::new();
    let out = project_batch(&batch, items, &mut scratch).unwrap();

    // Schema e lunghezze
    assert_eq!(out.schema, vec!["x", "y", "mask_10pct"]);
    assert_eq!(out.len, batch.len);
    assert_eq!(out.cols.len(), out.schema.len());

    // Ogni colonna deve essere identica allo slice d’ingresso
    for j in 0..out.cols.len() {
        match (&batch.cols[j], &out.cols[j]) {
            (ColumnView::I64(src), Column::I64(dst)) => assert_eq!(src, &dst.as_slice()),
            (ColumnView::Bool(src), Column::Bool(dst)) => assert_eq!(src, &dst.as_slice()),
            _ => panic!("tipo colonna non corrispondente"),
        }
    }
}

#[test]
fn project_keep_with_alias_and_order() {
    let t = build_table(10, 7);
    let mut scan = Scan::with_chunk(&t, 6);
    let batch = scan.next_batch().unwrap();

    // Keep con alias e ri-ordine (y prima di x)
    let items = &[
        ProjItem::Keep {
            name: "y",
            alias: Some("y_dup"),
        },
        ProjItem::Keep {
            name: "x",
            alias: None,
        },
    ];

    let mut scratch = ProjectScratch::new();
    let out = project_batch(&batch, items, &mut scratch).unwrap();

    assert_eq!(out.schema, vec!["y_dup", "x"]);
    assert_eq!(out.len, batch.len);
    assert_eq!(out.cols.len(), 2);

    // y → y_dup
    match (&batch.cols[1], &out.cols[0]) {
        (ColumnView::I64(src), Column::I64(dst)) => assert_eq!(src, &dst.as_slice()),
        _ => panic!("y attesa I64"),
    }
    // x → x
    match (&batch.cols[0], &out.cols[1]) {
        (ColumnView::I64(src), Column::I64(dst)) => assert_eq!(src, &dst.as_slice()),
        _ => panic!("x attesa I64"),
    }
}

#[test]
fn project_compute_only_add_and_gt() {
    let t = build_table(20, 99);
    let mut scan = Scan::with_chunk(&t, 13);
    let batch = scan.next_batch().unwrap();
    let n = batch.len;

    let items = &[
        ProjItem::ComputeI64 {
            name: "x_plus_1",
            expr: Expr::Add(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(1))),
        },
        ProjItem::ComputeBool {
            name: "x_gt_500k",
            expr: Expr::Gt(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(500_000))),
        },
    ];

    let mut scratch = ProjectScratch::new();
    let out = project_batch(&batch, items, &mut scratch).unwrap();

    assert_eq!(out.schema, vec!["x_plus_1", "x_gt_500k"]);
    assert_eq!(out.len, n);
    assert_eq!(out.cols.len(), 2);

    // Verità-terra
    let x_slice = match &batch.cols[0] {
        ColumnView::I64(s) => s,
        _ => panic!(),
    };
    let truth_add: Vec<i64> = x_slice.iter().map(|&x| x + 1).collect();
    let truth_gt: Vec<bool> = x_slice.iter().map(|&x| x > 500_000).collect();

    match &out.cols[0] {
        Column::I64(v) => assert_eq!(v, &truth_add),
        _ => panic!("x_plus_1 attesa I64"),
    }
    match &out.cols[1] {
        Column::Bool(v) => assert_eq!(v, &truth_gt),
        _ => panic!("x_gt_500k attesa Bool"),
    }
}

#[test]
fn project_mix_keep_and_compute_with_alias() {
    let t = build_table(15, 1234);
    let mut scan = Scan::with_chunk(&t, 8);
    let batch = scan.next_batch().unwrap();
    let n = batch.len;

    let items = &[
        ProjItem::Keep {
            name: "y",
            alias: Some("y_dup"),
        },
        ProjItem::ComputeI64 {
            name: "x2",
            expr: Expr::Add(Box::new(Expr::Col("x")), Box::new(Expr::Col("x"))),
        },
    ];

    let mut scratch = ProjectScratch::new();
    let out = project_batch(&batch, items, &mut scratch).unwrap();

    assert_eq!(out.schema, vec!["y_dup", "x2"]);
    assert_eq!(out.len, n);
    assert_eq!(out.cols.len(), 2);

    // y -> y_dup
    match (&batch.cols[1], &out.cols[0]) {
        (ColumnView::I64(src), Column::I64(dst)) => assert_eq!(src, &dst.as_slice()),
        _ => panic!("y attesa I64"),
    }
    // x2 = x + x
    let x_slice = match &batch.cols[0] {
        ColumnView::I64(s) => s,
        _ => panic!(),
    };
    let truth_x2: Vec<i64> = x_slice.iter().map(|&x| x + x).collect();
    match &out.cols[1] {
        Column::I64(v) => assert_eq!(v, &truth_x2),
        _ => panic!("x2 attesa I64"),
    }
}

#[test]
fn project_errors_unknown_and_typemismatch() {
    let t = build_table(8, 55);
    let mut scan = Scan::with_chunk(&t, 5);
    let batch = scan.next_batch().unwrap();

    let mut scratch = ProjectScratch::new();

    // Keep su colonna inesistente
    let items_unknown = &[ProjItem::Keep {
        name: "no_such_col",
        alias: None,
    }];
    let res = project_batch(&batch, items_unknown, &mut scratch);
    assert!(matches!(res, Err(EvalError::UnknownColumn(_))));

    // ComputeI64 con expr booleana -> TypeMismatch
    let items_bad_i64 = &[ProjItem::ComputeI64 {
        name: "oops",
        expr: Expr::Gt(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(0))),
    }];
    let res2 = project_batch(&batch, items_bad_i64, &mut scratch);
    assert!(matches!(res2, Err(EvalError::TypeMismatch(_))));

    // ComputeBool con expr numerica -> TypeMismatch
    let items_bad_bool = &[ProjItem::ComputeBool {
        name: "oops_b",
        expr: Expr::Add(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(1))),
    }];
    let res3 = project_batch(&batch, items_bad_bool, &mut scratch);
    assert!(matches!(res3, Err(EvalError::TypeMismatch(_))));
}

#[test]
fn project_across_all_batches_keep_concat_matches_table() {
    // Verifica che project non cambi il numero di righe e mantenga l’ordine su T intera
    let n = 30_000;
    let chunk = 4096;
    let t = build_table(n, 777);
    let (x_full, y_full, m_full) = match (&t.cols[0], &t.cols[1], &t.cols[2]) {
        (Column::I64(a), Column::I64(b), Column::Bool(c)) => (a, b, c),
        _ => panic!(),
    };

    let items = &[
        ProjItem::Keep {
            name: "x",
            alias: None,
        },
        ProjItem::Keep {
            name: "y",
            alias: None,
        },
        ProjItem::Keep {
            name: "mask_10pct",
            alias: None,
        },
    ];

    let mut scan = Scan::with_chunk(&t, chunk);
    let mut scratch = ProjectScratch::new();

    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut ms = Vec::with_capacity(n);

    while let Some(batch) = scan.next_batch() {
        let out = project_batch(&batch, items, &mut scratch).unwrap();
        // accoda in ordine
        match &out.cols[0] {
            Column::I64(v) => xs.extend_from_slice(v),
            _ => panic!(),
        }
        match &out.cols[1] {
            Column::I64(v) => ys.extend_from_slice(v),
            _ => panic!(),
        }
        match &out.cols[2] {
            Column::Bool(v) => ms.extend_from_slice(v),
            _ => panic!(),
        }
    }

    assert_eq!(&xs, x_full);
    assert_eq!(&ys, y_full);
    assert_eq!(&ms, m_full);
}

/* ---------- TEST: Milestone 5 ---------- */

#[test]
fn pipeline_mask_vs_indices_equivalence() {
    let t = build_table(80_000, 2025);
    let items = vec![
        ProjItem::Keep {
            name: "y",
            alias: Some("y_dup"),
        },
        ProjItem::ComputeI64 {
            name: "x_plus_1",
            expr: Expr::Add(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(1))),
        },
        ProjItem::ComputeBool {
            name: "hot",
            expr: Expr::Gt(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(500_000))),
        },
    ];
    let pred = Expr::Gt(Box::new(Expr::Col("x")), Box::new(Expr::LitI64(200_000)));

    let mut p_m = Pipeline::with_chunk(&t, 4096, pred.clone(), FilterStrategy::Mask, items.clone());
    let mut p_i = Pipeline::with_chunk(&t, 4096, pred, FilterStrategy::Indices, items);

    loop {
        let bm = p_m.next().unwrap();
        let bi = p_i.next().unwrap();

        match (bm, bi) {
            (None, None) => break,
            (Some(bm), Some(bi)) => {
                // stesse righe e stesso schema
                assert_eq!(bm.len, bi.len);
                assert_eq!(bm.schema, bi.schema);
                assert_eq!(bm.cols.len(), bi.cols.len());

                // colonne identiche element-wise
                for j in 0..bm.cols.len() {
                    match (&bm.cols[j], &bi.cols[j]) {
                        (Column::I64(a), Column::I64(b)) => assert_eq!(a, b),
                        (Column::Bool(a), Column::Bool(b)) => assert_eq!(a, b),
                        _ => panic!("tipo colonna non corrispondente"),
                    }
                }
            }
            _ => panic!("le due pipeline hanno prodotto numeri diversi di batch"),
        }
    }
}
