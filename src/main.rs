mod optimal_dft;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use image::{GrayImage, Luma};

use optimal_dft::get_optimal_dft_size;
use std::time::Instant;
use rayon::prelude::*;
use std::sync::Arc;
use rand::Rng;

type VecCpx32 = Vec<Complex32>;
type FftF32 = dyn Fft<f32>;

const PSR: f32 = 5.7;
const EPS: f32 = 1e-5;
const WARP: f32 = 0.1;
const LEARNRATE: f32 = 0.2;
const EPSCPX: Complex32 = Complex32::new(EPS, 0f32);

struct Precomputed {
    fft: Arc<FftF32>,
    ifft: Arc<FftF32>,
    gaussian: VecCpx32,
    hann: Vec<f32>,
    size: usize,
    area: usize,
    cx: f32,
}

impl Precomputed {
    fn new(planner: &mut FftPlanner<f32>, size: usize) -> Self {
        let nm = (size - 1) as f32;
        let area = size * size;
        let cx = nm / 2f32;
        let cy = nm / 2f32;
        
        // Hann window tapering function
        // Combats Heisenberg uncertainty
        let mut hann = Vec::with_capacity(area);
        let pi2 = 2f32 * std::f32::consts::PI;

        // Represents an ideal response
        // Peaks at center of bounding box
        let mut gaussian = Vec::with_capacity(area);
        let ifft = planner.plan_fft_inverse(size);
        let fft = planner.plan_fft_forward(size);
        
        for i in 0..size {
            // Optimization: compute this once per row
            let wy = 1f32 - (pi2 * i as f32 / nm).cos(/**/);
            let dy = i as f32 - cy;
            
            for j in 0..size {
                let dx = j as f32 - cx;
                let result = (-0.5 * (dx * dx + dy * dy) / 4f32).exp(/**/);
                let wx = 1f32 - (pi2 * j as f32 / nm).cos(/**/);
                hann.push(wy * wx / 4f32);
                gaussian.push(result);
            }
        }

        let max = gaussian.iter(/**/).cloned(/**/).fold(f32::MIN, f32::max);
        gaussian.iter_mut(/**/).for_each(|value| *value /= max);

        let mut gaussian = to_complex(&gaussian, 0f32);
        fft2dd(&mut gaussian, &fft, 1f32, size);

        Precomputed { fft, ifft, gaussian, hann, size, area, cx }
    }
}

struct Cache {
    planner: FftPlanner<f32>,
    cache: Vec<Precomputed>,
}

impl Cache {
    fn get(&mut self, size: usize) -> &Precomputed {
        if let Some(idx) = self.cache.iter(/**/).position(|pre| pre.size == size) {
            // Internally we only do square bounding boxes to simplify things
            // Among other perks this allows for simple caching mechanism
            return &self.cache[idx];
        }
        
        let fresh_precomputed = Precomputed::new(&mut self.planner, size);
        self.cache.push(fresh_precomputed);
        self.cache.last(/**/).unwrap(/**/)
    }
}

struct Mosse<'a> {
    pre: &'a Precomputed,
    scratch: VecCpx32,
    h: VecCpx32,
    a: VecCpx32,
    b: VecCpx32,
    cx: f32,
    cy: f32,
}

impl<'a> Mosse<'a> {
    pub fn new(cache: &'a mut Cache, img: &GrayImage, cx: f32, cy: f32, raw_size: usize) -> Self {
        let size = get_optimal_dft_size(raw_size);
        let pre = cache.get(size);

        let patch = crop(img, pre, cx, cy);
        let parts: Vec<(VecCpx32, VecCpx32)> =
            (0..8).into_par_iter(/**/).map(|_| {
                let mut target = vec![0f32; pre.area];
                warp(&patch, &mut target, pre.cx, pre.cx, size);
                preprocess(&mut target, &pre.hann, 0f32);

                let mut target = to_complex(&target, 0f32);
                fft2d(&mut target, pre, false);

                let mut apass = vec![Complex32::ZERO; pre.area];
                let mut bpass = vec![Complex32::ZERO; pre.area];

                for idx in 0..pre.area {
                    apass[idx] = pre.gaussian[idx] * target[idx].conj(/**/);
                    bpass[idx] = target[idx] * target[idx].conj(/**/);
                }

                (apass, bpass)
            }).collect(/**/);

        let mut a = vec![Complex32::ZERO; pre.area];
        let mut b = vec![Complex32::ZERO; pre.area];
        let aiter = a.par_iter_mut(/**/);
        let biter = b.par_iter_mut(/**/);

        aiter.zip(biter).enumerate(/**/).for_each(|(i, ab)| {
            let mut asum = Complex32::ZERO;
            let mut bsum = Complex32::ZERO;
            
            for (ai, bi) in &parts {
                asum += ai[i];
                bsum += bi[i];
            }

            *ab.0 = asum;
            *ab.1 = bsum;
        });

        let aiter = a.iter(/**/);
        let biter = b.iter(/**/);

        let h = aiter.zip(biter).map(|(a, b)| {
            // Build an actual correlation filter
            *a / (*b + EPSCPX)
        }).collect(/**/);
        
        let scratch = vec![Complex32::ZERO; pre.area];
        Mosse { pre, scratch, h, a, b, cx, cy }
    }

    pub fn update(&mut self, img: &GrayImage) -> Option<(f32, f32)> {
        let current = self.extract_and_fft(img);
    
        for i in 0..self.pre.area {
            let filter = self.h[i].conj(/**/);
            self.scratch[i] = current[i] * filter;
        }

        fft2d(&mut self.scratch, self.pre, true);
    
        let mut sum = 0f32;
        let mut sumsq = 0f32;
        let mut maxv = f32::MIN;
        let mut idx = 0;

        for (i, c) in self.scratch.iter(/**/).enumerate(/**/) {
            // Compute peak-to-sidelobe ratio in frequency domain
            sumsq += c.re * c.re;
            sum += c.re;

            if c.re > maxv {
                maxv = c.re;
                idx = i;
            }
        }

        let mean = sum / self.pre.area as f32;
        let var = sumsq / self.pre.area as f32 - mean * mean;
        if (maxv - mean) / var.sqrt(/**/).max(EPS) < PSR {
            return None;
        }
    
        let row = idx / self.pre.size;
        let col = idx % self.pre.size;
        self.cx += col as f32 - (self.pre.size as f32 / 2f32);
        self.cy += row as f32 - (self.pre.size as f32 / 2f32);
        
        let result = (self.cx, self.cy);
        let next = self.extract_and_fft(img);

        for i in 0..self.pre.area {
            let an = self.h[i] * next[i].conj(/**/) * LEARNRATE;
            let bn = next[i] * next[i].conj(/**/) * LEARNRATE;
            self.a[i] = self.a[i] * (1f32 - LEARNRATE) + an;
            self.b[i] = self.b[i] * (1f32 - LEARNRATE) + bn;
            self.h[i] = self.a[i] / (self.b[i] + EPSCPX);
        }
        
        Some(result)
    }

    #[inline]
    fn extract_and_fft(&self, img: &GrayImage) -> VecCpx32 {
        let mut buf = crop(img, self.pre, self.cx, self.cy);
        preprocess(&mut buf, &self.pre.hann, 0f32);

        let mut buf = to_complex(&buf, 0f32);
        fft2d(&mut buf, self.pre, false);
        buf
    }
}

#[inline]
fn fft2d(buf: &mut [Complex32], pre: &Precomputed, ifft: bool) {
    let scale = if ifft { 1f32 / pre.area as f32 } else { 1f32 };
    let fourier = if ifft { &pre.ifft } else { &pre.fft };
    fft2dd(buf, fourier, scale, pre.size);
}

#[inline]
fn fft2dd(buf: &mut [Complex32], fourier: &Arc<FftF32>, scale: f32, n: usize) {
    // Applies 2D (row-wise, then column-wise) FFT/IFFT, avoids full transpose
    buf.par_chunks_mut(n).for_each(|value| {
        fourier.process(value);
    });

    for j in 0..n {
        // This factually builds a column
        let mut col = Vec::with_capacity(n);

        for i in 0..n { 
            let cell = buf[i * n + j];
            col.push(cell); 
        }

        fourier.process(&mut col);

        for i in 0..n { 
            let cell = col[i] * scale;
            buf[i * n + j] = cell;
        }
    }
}

#[inline]
fn crop(img: &GrayImage, pre: &Precomputed, cx: f32, cy: f32) -> Vec<f32> {
    let shift = (pre.size as f32 - 1f32) / 2f32;
    let mut out = vec![0f32; pre.area];
    let h = img.height(/**/) as isize;
    let w = img.width(/**/) as isize;

    out.par_chunks_mut(pre.size).enumerate(/**/).for_each(|(i, row)| {
           let yf = cy + (i as f32 - shift);
           let y0 = yf.floor(/**/) as isize;
           let dy = yf - y0 as f32;
           let wy0 = 1f32 - dy;

           let y00 = reflect(y0, h);
           let y01 = reflect(y0 + 1, h);
           
           for (j, cell) in row.iter_mut(/**/).enumerate(/**/) {
                let xf = cx + (j as f32 - shift);
                let x0 = xf.floor(/**/) as isize;
                let dx = xf - x0 as f32;
                let wx0 = 1f32 - dx;
                let wx1 = dx;
                
                let x00 = reflect(x0, w);
                let x10 = reflect(x0 + 1, w);

                let i00 = img.get_pixel(x00 as u32, y00 as u32)[0] as f32 + 1f32;
                let i10 = img.get_pixel(x10 as u32, y00 as u32)[0] as f32 + 1f32;
                let i01 = img.get_pixel(x00 as u32, y01 as u32)[0] as f32 + 1f32;
                let i11 = img.get_pixel(x10 as u32, y01 as u32)[0] as f32 + 1f32;
                *cell = i00 * wx0 * wy0 + i10 * wx1 * wy0 + i01 * wx0 * dy + i11 * wx1 * dy;
           }
       });

    out
}

#[inline]
fn preprocess(buf: &mut [f32], hann: &[f32], def: f32) {
    let len = buf.len(/**/) as f32;
    let mut sum = def;
    let mut sq = def;

    for value in buf.iter_mut(/**/) {
        *value = value.ln(/**/);
        sq += *value * *value;
        sum += *value;
    }

    let mean = sum / len;
    let var = sq / len - mean * mean;
    let std = var.sqrt(/**/) + EPS;

    let buf_iter = buf.iter_mut(/**/);
    let hann_iter = hann.iter(/**/);

    // Log-normalize and apply Hann window
    for (v, &w) in buf_iter.zip(hann_iter) {
        *v = (*v - mean) / std * w;
    }
}

#[inline]
fn to_complex(rc: &[f32], im: f32) -> VecCpx32 {
    let to_pseudo_complex = |val: &f32| Complex32::new(*val, im);
    rc.iter(/**/).map(to_pseudo_complex).collect(/**/)
}

#[inline]
fn warp(src: &[f32], dst: &mut [f32], cx: f32, cy: f32, n: usize) {
    // Applies random rotation + shear + scale to a given patch
    let mut rng = rand::rng(/**/);
    
    let ang = rng.random_range(-WARP..WARP);
    let c = ang.cos(/**/);
    let s = ang.sin(/**/);

    let w00 = c + rng.random_range(-WARP..WARP);
    let w01 = -s + rng.random_range(-WARP..WARP);
    let w10 = s + rng.random_range(-WARP..WARP);
    let w11 = c + rng.random_range(-WARP..WARP);

    let w02 = cx - (w00 * cx + w01 * cy);
    let w12 = cy - (w10 * cx + w11 * cy);

    for i in 0..n as isize {
        for j in 0..n as isize {
            let u = w00 * j as f32 + w01 * i as f32 + w02;
            let v = w10 * j as f32 + w11 * i as f32 + w12;

            // integer base + fraction
            let x0 = u.floor(/**/) as isize;
            let y0 = v.floor(/**/) as isize;
            let dx = u - x0 as f32;
            let dy = v - y0 as f32;

            // reflect101 on four corners
            let x00 = reflect(x0, n as isize) as usize;
            let y00 = reflect(y0, n as isize) as usize;
            let x10 = reflect(x0 + 1, n as isize) as usize;
            let y01 = reflect(y0 + 1, n as isize) as usize;

            let i00 = src[y00 * n + x00];
            let i10 = src[y00 * n + x10];
            let i01 = src[y01 * n + x00];
            let i11 = src[y01 * n + x10];

            // bilinear interpolate
            let t0 = i00 * (1.0 - dx) + i10 * dx;
            let t1 = i01 * (1.0 - dx) + i11 * dx;
            let index = (i as usize) * n + (j as usize);
            dst[index] = t0 * (1.0 - dy) + t1 * dy;
        }
    }
}

#[inline]
fn reflect(idx: isize, len: isize) -> usize {
    // Mirror‐border reflect with 101 behavior
    let mut x = idx;

    while x < 0 || x >= len {
        x = if x < 0 {
            -x
        } else {
            2 * len - x
        };
    }
    
    x as usize
}

fn main(/**/) {
    let mut cache = Cache { 
        planner: FftPlanner::new(/**/),
        cache: Vec::new(/**/)
    };

    let img = GrayImage::from_pixel(800, 800, Luma([128]));
    let raw_size = 640;
    let x0 = 80f32;
    let y0 = 80f32;

    let now = Instant::now();
    let _ = Mosse::new(&mut cache, &img, x0, y0, raw_size);
    println!("init time cold: {:?}", now.elapsed(/**/));

    let now = Instant::now();
    let mut mosse = Mosse::new(&mut cache, &img, x0, y0, raw_size);
    println!("init time warm: {:?}", now.elapsed(/**/));

    let now = Instant::now();
    let result = mosse.update(&img);
    println!("update time: {:?}", now.elapsed(/**/));
    println!("result: {:?}", result);
}