pub const MAX: usize = 360;
pub const MIN: usize = 10;

const SIZES: &[usize] = &[
    MIN, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
    50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 
    128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 
    243, 250, 256, 270, 288, 300, 320, 324, MAX,
];

pub const LEN: usize = SIZES.len(/**/);

pub fn get_size(size: usize) -> usize {
    if size <= MIN { return MIN; }
    if size >= MAX { return MAX; }

    let idx = SIZES
        .binary_search(&size)
        .unwrap_or_else(|idx| idx);

    SIZES[idx]
}
