static OPTIMAL_DFT_SIZE_TAB: &[usize] = &[
    1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
    50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160,
    162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360
];

pub static LEN: usize = OPTIMAL_DFT_SIZE_TAB.len(/**/);

pub fn get_size(size: usize) -> usize {
    match OPTIMAL_DFT_SIZE_TAB.binary_search(&size) {
        Err(index) if index < LEN => OPTIMAL_DFT_SIZE_TAB[index],
        Ok(index) => OPTIMAL_DFT_SIZE_TAB[index],
        _ => 360
    }
}