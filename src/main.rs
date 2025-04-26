#![feature(portable_simd)]

use std::f32::{self, consts::PI};
use std::simd::StdFloat;
use std::sync::Arc;
use image::{GrayImage, Luma};
use optimal_dft::get_optimal_dft_size;
use rand::Rng;
use rustfft::num_traits::ConstZero;
use rustfft::{num_complex::Complex32, Fft, FftPlanner, FftPlannerNeon};
use rayon::prelude::*;

use core::simd::Simd;
use core::simd::prelude::SimdFloat;
mod optimal_dft;

const LANES: usize = 4;
type Vf = Simd<f32, LANES>;

pub struct Mosse {
    h: Vec<Complex32>,
    a: Vec<Complex32>,
    b: Vec<Complex32>,
    hann: Vec<f32>,
    size: usize,
    cx: f32,
    cy: f32,
    fft: Arc<dyn Fft<f32> + Sync + Send>,
    ifft: Arc<dyn Fft<f32> + Sync + Send>,
    scratch: Vec<Complex32>,
}

impl Mosse {
    pub fn new(
        planner: &mut FftPlannerNeon<f32>,
        img: &GrayImage,
        x: u32,
        y: u32,
        size: usize,
    ) -> Self {
        // enforce square
        let n = get_optimal_dft_size(size);
        let cx = (n - 1) as f32 * 0.5;
        let cy = cx;

        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        let hann = make_hann(n);
        let gaussian = make_gaussian(n, cx, cy);

        let patch0 = crop(img, x, y, n as u32, n as u32);

        let mut G = to_complex(&gaussian);
        fft2d(&*fft, n, &mut G, false);

        let partials: Vec<(Vec<Complex32>, Vec<Complex32>)> =
            (0..8).into_par_iter().map(|_| {
                let mut p = vec![0.0; n * n];
                random_warp(&patch0, &mut p, cx, cy, n);
                preprocess(&mut p, &hann);
                let mut F = to_complex(&p);
                fft2d(&*fft, n, &mut F, false);
                let mut Ai = vec![Complex32::ZERO; n * n];
                let mut Bi = vec![Complex32::ZERO; n * n];
                for idx in 0..n * n {
                    Ai[idx] = G[idx] * F[idx].conj();
                    Bi[idx] = F[idx] * F[idx].conj();
                }
                (Ai, Bi)
            }).collect();

        let mut A = vec![Complex32::ZERO; n * n];
        let mut B = vec![Complex32::ZERO; n * n];
        A.par_iter_mut().zip(B.par_iter_mut()).enumerate().for_each(|(i, (ai, bi))| {
            let mut sumA = Complex32::ZERO;
            let mut sumB = Complex32::ZERO;
            for (Ai, Bi) in &partials {
                sumA += Ai[i];
                sumB += Bi[i];
            }
            *ai = sumA;
            *bi = sumB;
        });

        let eps = Complex32::new(1e-5, 0.0);
        let H = A.iter().zip(B.iter()).map(|(a,b)| *a / (*b + eps)).collect();
        let scratch = vec![Complex32::ZERO; n * n];

        Mosse { h: H, a: A, b: B, hann, size: n, cx, cy, fft, ifft, scratch }
    }
}

#[inline]
fn make_hann(n: usize) -> Vec<f32> {
    // Brings 2D signal edges to zeros
    // Thus combats Heisenberg uncertainty
    let mut window = Vec::with_capacity(n * n);
    let pi2 = 2f32 * f32::consts::PI;
    let nm = (n - 1) as f32;

    for i in 0..n {
        // Optimization: compute this once per row
        let wy = 1f32 - (pi2 * i as f32 / nm).cos(/**/);
        
        for j in 0..n {
            let wx = 1f32 - (pi2 * j as f32 / nm).cos(/**/);
            window.push(wy * wx / 4f32);
        }
    }

    window
}

#[inline]
fn make_gaussian(n: usize, cx: f32, cy: f32) -> Vec<f32> {
    // Constructs an "ideal response" gaussian with peak at center
    let mut gaussian = Vec::with_capacity(n * n);
    let sigma = 4f32;

    for i in 0..n {
        for j in 0..n {
            let dy = i as f32 - cy;
            let dx = j as f32 - cx;
            let cell = -0.5 * (dx * dx + dy * dy);
            let result = (cell / sigma).exp(/**/);
            gaussian.push(result);
        }
    }

    let max = gaussian.iter(/**/).cloned(/**/).fold(f32::MIN, f32::max);
    gaussian.iter_mut(/**/).for_each(|val| { *val /= max });
    gaussian
}

#[inline]
fn crop(img: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> Vec<f32> {
    let patch = image::imageops::crop_imm(img, x, y, w, h).to_image(/**/);
    patch.pixels(/**/).map(|buf| { buf[0] as f32 + 1.0 }).collect(/**/)
}

#[inline]
fn preprocess(buf: &mut [f32], hann: &[f32]) {
    let mut sumsq_simd = Vf::splat(0f32);
    let mut sum_simd = Vf::splat(0f32);
    let len = buf.len(/**/) as f32;

    // Log-normalize
    for chunk in buf.chunks_mut(LANES) {
        let val = Vf::from_slice(chunk).ln(/**/);
        sumsq_simd += val * val;
        sum_simd += val;
        
        let arr = &val.to_array(/**/);
        chunk.copy_from_slice(arr);
    }

    let mean = sum_simd.reduce_sum(/**/) / len;
    let var = sumsq_simd.reduce_sum(/**/) / len - mean * mean;
    let std_splat = Vf::splat(var.sqrt(/**/) + 1e-5);
    let mean_splat = Vf::splat(mean);
    let pl = buf.chunks_mut(LANES);
    let hl = hann.chunks(LANES);

    // Apply hann window
    for (pc, hc) in pl.zip(hl) {
        let mut val = Vf::from_slice(pc);
        val = (val - mean_splat) / std_splat;
        val *= Vf::from_slice(hc);

        let arr = &val.to_array(/**/);
        pc.copy_from_slice(arr);
    }
}

#[inline]
fn to_complex(rc: &[f32]) -> Vec<Complex32> {
    rc.iter(/**/).map(|&val| { Complex32::new(val, 0f32) }).collect(/**/)
}

#[inline]
fn fft2d(fft: &dyn Fft<f32>, n: usize, buf: &mut [Complex32], ifft: bool) {
    // Applies 2D (row-wise, then column-wise) FFT/IFFT, avoids full transpose
    let scale = if ifft { 1f32 / (n * n) as f32 } else { 1f32 };
    buf.par_chunks_mut(n).for_each(|val| { fft.process(val) });

    for j in 0..n {
        // This factually builds a column
        let mut col = Vec::with_capacity(n);

        for i in 0..n { 
            let cell = buf[i * n + j];
            col.push(cell); 
        }

        fft.process(&mut col);

        for i in 0..n { 
            let cell = col[i] * scale;
            buf[i * n + j] = cell;
        }
    }
}

#[inline]
fn random_warp(src: &[f32], dst: &mut [f32], cx: f32, cy: f32, n: usize) {
    // Applies random rotation + shear + scale to a given patch
    let mut rng = rand::rng(/**/);
    let warp = 0.1f32;

    let ang = rng.random_range(-warp..warp);
    let c = ang.cos(/**/);
    let s = ang.sin(/**/);

    let w00 = c + rng.random_range(-warp..warp);
    let w01 = -s + rng.random_range(-warp..warp);
    let w10 = s + rng.random_range(-warp..warp);
    let w11 = c + rng.random_range(-warp..warp);

    let w02 = cx - (w00 * cx + w01 * cy);
    let w12 = cy - (w10 * cx + w11 * cy);

    for i in 0..n as isize {
        for j in 0..n as isize {
            let u = w00 * j as f32 + w01 * i as f32 + w02;
            let v = w10 * j as f32 + w11 * i as f32 + w12;
            let ui = reflect(u.round(/**/) as isize, n as isize);
            let vi = reflect(v.round(/**/) as isize, n as isize);
            let index = (i as usize) * n + (j as usize);
            dst[index] = src[vi * n + ui];
        }
    }
}

#[inline]
fn reflect(idx: isize, len: isize) -> usize {
    // Same as BORDER_REFLECT in opencv
    let mut x = idx;

    while x < 0 || x >= len {
        x = if x < 0 { -x - 1 } else { 2 * len - x - 1 };
    }

    x as usize
}

fn main() {
    let mut planner = FftPlannerNeon::<f32>::new().expect("Failed to create planner");
    let img = GrayImage::from_pixel(800,800,Luma([128]));
    let _ = Mosse::new(&mut planner, &img, 80, 80, 375);
    let now = std::time::Instant::now();
    let _ = Mosse::new(&mut planner, &img, 80, 80, 375);
    println!("initialization time: {:?}", now.elapsed());
}
