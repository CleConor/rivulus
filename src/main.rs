use rivulus::{Column, ColumnView, Scan, Table};

fn main() {
    println!("== Case 1: N=10, chunk=4 ==");
    let t1 = build_table(10, 42);
    run_scan_checks(&t1, 4, /*print_samples=*/ true);

    println!("\n== Case 2: N=8192, chunk=8192 ==");
    let t2 = build_table(8192, 123);
    run_scan_checks(&t2, 8192, /*print_samples=*/ false);

    println!("\n== Case 3: N=8193, chunk=8192 ==");
    let t3 = build_table(8193, 999);
    run_scan_checks(&t3, 8192, /*print_samples=*/ false);

    println!("\nTutti i check del main sono passati ✅");
}

/* -------------------- Helpers di test/demo -------------------- */

fn run_scan_checks(table: &Table, chunk: usize, print_samples: bool) {
    // Esegui lo scan e raccogli batch.len
    let (lengths, x_full, y_full, m_full) = scan_collect(table, chunk);

    // Verifica che la concatenazione dei batch ricostruisca le colonne originali
    let (x_orig, y_orig, m_orig) = borrow_table_slices(table);
    assert_eq!(x_full, x_orig, "x ricostruita non combacia con l'originale");
    assert_eq!(y_full, y_orig, "y ricostruita non combacia con l'originale");
    assert_eq!(
        m_full, m_orig,
        "mask ricostruita non combacia con l'originale"
    );

    // Verifica pattern delle lunghezze (tutti 'chunk' tranne l'ultimo)
    let expected_batches = (table.len + chunk - 1) / chunk;
    assert_eq!(lengths.len(), expected_batches, "numero batch inatteso");
    for (i, &len) in lengths.iter().enumerate() {
        let is_last = i + 1 == expected_batches;
        if !is_last {
            assert_eq!(len, chunk, "batch {} deve avere len=chunk", i);
        } else {
            let expected_last = if table.len % chunk == 0 {
                chunk
            } else {
                table.len % chunk
            };
            assert_eq!(len, expected_last, "ultimo batch ha len errata");
        }
    }

    println!("batch lengths = {:?}", lengths);

    if print_samples && !lengths.is_empty() {
        // stampa 3 elementi del primo batch e 3 dell'ultimo
        let first_n = 3.min(x_full.len());
        let last_n = 3.min(x_full.len());

        println!("prime {} righe (x, mask):", first_n);
        for i in 0..first_n {
            println!("  {} {}", x_full[i], m_full[i]);
        }
        println!("ultime {} righe (x, mask):", last_n);
        for i in (x_full.len() - last_n)..x_full.len() {
            println!("  {} {}", x_full[i], m_full[i]);
        }
    }
}

fn scan_collect(table: &Table, chunk: usize) -> (Vec<usize>, Vec<i64>, Vec<i64>, Vec<bool>) {
    let mut scan = Scan::with_chunk(table, chunk);

    let mut lengths = Vec::new();
    let mut x_all = Vec::with_capacity(table.len);
    let mut y_all = Vec::with_capacity(table.len);
    let mut m_all = Vec::with_capacity(table.len);

    while let Some(batch) = scan.next_batch() {
        // verifica invarianti sul batch
        debug_assert!(batch.cols.iter().all(|c| c.len() == batch.len));

        lengths.push(batch.len);
        // col 0: x, col 1: y, col 2: mask (bool)
        match &batch.cols[0] {
            ColumnView::I64(s) => x_all.extend_from_slice(s),
            _ => panic!("colonna 0 deve essere I64"),
        }
        match &batch.cols[1] {
            ColumnView::I64(s) => y_all.extend_from_slice(s),
            _ => panic!("colonna 1 deve essere I64"),
        }
        match &batch.cols[2] {
            ColumnView::Bool(s) => m_all.extend_from_slice(s),
            _ => panic!("colonna 2 deve essere Bool"),
        }
    }

    (lengths, x_all, y_all, m_all)
}

fn borrow_table_slices<'a>(table: &'a Table) -> (&'a [i64], &'a [i64], &'a [bool]) {
    let x = match &table.cols[0] {
        Column::I64(v) => v.as_slice(),
        _ => panic!("cols[0] deve essere I64"),
    };
    let y = match &table.cols[1] {
        Column::I64(v) => v.as_slice(),
        _ => panic!("cols[1] deve essere I64"),
    };
    let m = match &table.cols[2] {
        Column::Bool(v) => v.as_slice(),
        _ => panic!("cols[2] deve essere Bool"),
    };
    (x, y, m)
}

/* -------------------- Generatore tabella sintetica -------------------- */

fn build_table(n: usize, seed: u64) -> Table {
    // LCG deterministico (niente dipendenze esterne)
    let mut s1 = seed ^ 0x9E3779B97F4A7C15;
    let mut s2 = seed.wrapping_add(0xBF58476D1CE4E5B9);

    fn lcg(state: &mut u64) -> u64 {
        // classico LCG 64-bit
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut mask = Vec::with_capacity(n);

    const M: i64 = 1_000_000; // range [0, M)
    for _ in 0..n {
        let xi = (lcg(&mut s1) % (M as u64)) as i64;
        let yi = (lcg(&mut s2) % (M as u64)) as i64;
        let keep = xi % 10 == 0; // maschera derivata: ~10% true
        x.push(xi);
        y.push(yi);
        mask.push(keep);
    }

    // Invarianti: stessa lunghezza + schema fissato
    let schema: [&'static str; 3] = ["x", "y", "mask_10pct"];
    let cols = [Column::I64(x), Column::I64(y), Column::Bool(mask)];

    Table {
        schema,
        cols,
        len: n,
    }
}
