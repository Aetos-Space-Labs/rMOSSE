use std::f32::{self, consts::PI};
use std::sync::Arc;
use image::{GrayImage, Luma, imageops};
use optimal_dft::get_optimal_dft_size;
use rand::Rng;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use rayon::prelude::*;
mod optimal_dft;

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
        planner: &mut FftPlanner<f32>,
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

        let patch0 = crop(img, x, y, n as u32);

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

    pub fn update(&mut self, img: &GrayImage, cx: f32, cy: f32) -> Option<(f32, f32)> {
        // sample the patch → log-normalize & hann → FFT
        let mut p = crop(img, cx as u32, cy as u32, self.size as u32);
        preprocess(&mut p, &self.hann);
        let mut F = to_complex(&p);
        fft2d(&*self.fft, self.size, &mut F, false);
    
        // multiply by H in freq domain → IFFT
        let n2 = self.size * self.size;
        for i in 0..n2 {
            self.scratch[i] = F[i] * self.h[i];
        }
        fft2d(&*self.ifft, self.size, &mut self.scratch, true);
    
        // compute mean, variance, find peak index
        let mut sum = 0f32;
        let mut sumsq = 0f32;
        let mut maxv = f32::MIN;
        let mut idx = 0usize;
        for (i, c) in self.scratch.iter().enumerate() {
            let r = c.re;
            sum += r;
            sumsq += r * r;
            if r > maxv {
                maxv = r;
                idx = i;
            }
        }
        let n2f = n2 as f32;
        let mean = sum / n2f;
        let var = sumsq / n2f - mean * mean;
        let std = var.sqrt().max(1e-5);
        let psr = (maxv - mean) / std;
        if psr < 5.7 {
            return None;
        }
    
        // translate flat idx to (dx, dy) relative to patch center
        let row = idx / self.size;
        let col = idx % self.size;
        let dy = (row as i32) - (self.size as i32 / 2);
        let dx = (col as i32) - (self.size as i32 / 2);
    
        // new center
        let ncx = cx + dx as f32;
        let ncy = cy + dy as f32;
    
        // re-sample at the new center then preprocess then FFT
        let mut p2 = crop(img, ncx as u32, ncy as u32, self.size as u32);
        preprocess(&mut p2, &self.hann);
        let mut F2 = to_complex(&p2);
        fft2d(&*self.fft, self.size, &mut F2, false);
    
        // update A, B, and H
        let rate = 0.2f32;
        let eps  = Complex32::new(1e-5, 0.0);
        for i in 0..n2 {
            let An = self.h[i] * F2[i].conj();
            let Bn = F2[i] * F2[i].conj();
            self.a[i] = self.a[i] * (1.0 - rate) + An * rate;
            self.b[i] = self.b[i] * (1.0 - rate) + Bn * rate;
            self.h[i] = self.a[i] / (self.b[i] + eps);
        }
    
        Some((ncx, ncy))
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
fn crop(img: &GrayImage, x: u32, y: u32, s: u32) -> Vec<f32> {
    imageops::crop_imm(img, x, y, s, s).to_image(/**/).pixels(/**/).map(|luma| { 
        luma[0] as f32 + 1f32 
    }).collect(/**/)
}

#[inline]
fn preprocess(buf: &mut [f32], hann: &[f32]) {
    let len = buf.len(/**/) as f32;
    let mut sumsq = 0f32;
    let mut sum = 0f32;
    
    for value in buf.iter_mut(/**/) {
        *value = value.ln(/**/);
        sumsq += *value * *value;
        sum += *value;
    }

    let mean = sum / len;
    let var = sumsq / len - mean * mean;
    let std = var.sqrt(/**/) + 1e-5;

    let buf_iter = buf.iter_mut(/**/);
    let hann_iter = hann.iter(/**/);

    // Log-normalize and apply Hann window
    for (v, &w) in buf_iter.zip(hann_iter) {
        *v = (*v - mean) / std * w;
    }
}

#[inline]
fn to_complex(rc: &[f32]) -> Vec<Complex32> {
    rc.iter(/**/).map(|&val| { 
        Complex32::new(val, 0f32) 
    }).collect(/**/)
}

#[inline]
fn fft2d(fft: &dyn Fft<f32>, n: usize, buf: &mut [Complex32], ifft: bool) {
    // Applies 2D (row-wise, then column-wise) FFT/IFFT, avoids full transpose
    let scale = if ifft { 1f32 / (n * n) as f32 } else { 1f32 };

    buf.par_chunks_mut(n).for_each(|val| { 
        fft.process(val) 
    });

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
    let mut planner = FftPlanner::<f32>::new();
    let img = GrayImage::from_pixel(800,800,Luma([128]));
    let _ = Mosse::new(&mut planner, &img, 80, 80, 640);
    let now = std::time::Instant::now();
    let mut mosse = Mosse::new(&mut planner, &img, 80, 80, 640);
    println!("init time: {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let res = mosse.update(&img, 80f32, 80f32);
    println!("update time: {:?}", now.elapsed());
    println!("result: {:?}", res);
}
