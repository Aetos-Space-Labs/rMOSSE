use rand::Rng;
use rustfft::num_traits::Zero;
use rustfft::num_complex::{Complex, ComplexFloat};
use std::sync::Arc;
use std::{f32::consts::PI, time::Instant};
use rustfft::{FftPlanner, FftPlannerScalar};
use image::{imageops, GenericImageView, GrayImage, Luma};
use nalgebra::{ComplexField, DMatrix, DVector, RowDVector};
use rustfft::Fft;
use rayon::prelude::*;
mod optimal_dft;

const PI2: f32 = PI * 2f32;
const LEARNING_RATE: f32 = 0.2;
const NORMALIZATION: f32 = 1e-5;
const DETECTION_THRESHOLD: f32 = 5.7;
const TRAIN_TIMES: u8 = 8;
const WARP: f32 = 0.1;

type ComplexF32 = Complex<f32>;
type FftF32 = dyn Fft<f32>;

// Represents dimensions of patch to be tracked
// Has both top-left and centered coordinates

struct BoundingBox {
    length: usize,
    xcenter: f32,
    ycenter: f32,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl BoundingBox {
    pub fn new(x: usize, y: usize, w: usize, h: usize) -> Self {
        // Speed up FFT by finding a size breakable into small primes
        let w = optimal_dft::get_optimal_dft_size(w);
        let h = optimal_dft::get_optimal_dft_size(h);
        
        Self { 
            length: w * h, 
            xcenter: w as f32 / 2f32, 
            ycenter: h as f32 / 2f32, 
            x, y, w, h,
        }
    }
}

struct TrackerMOSSE {
    hann_window: DMatrix<f32>,
    h: DMatrix<ComplexF32>,
    a: DMatrix<ComplexF32>,
    b: DMatrix<ComplexF32>,
    g: DMatrix<ComplexF32>,
    fwd_fft: Arc<FftF32>,
    inv_fft: Arc<FftF32>,
    tlb: BoundingBox,
}

impl TrackerMOSSE {
    fn new(planner: &mut FftPlanner<f32>, tlb: BoundingBox) -> Self {
        let hann_col = DVector::from_iterator(tlb.h, (0..tlb.h).map(|y| {
            1.0 - (PI2 * y as f32 / (tlb.h - 1) as f32).cos(/**/) / 2f32
        }));

        let hann_row = RowDVector::from_iterator(tlb.w, (0..tlb.w).map(|x| {
            1.0 - (PI2 * x as f32 / (tlb.w - 1) as f32).cos(/**/) / 2f32
        }));
        
        let gx = RowDVector::from_iterator(tlb.w, (0..tlb.w).map(|x| {
            let real = -(x as f32 - tlb.xcenter).powi(2) / 2f32;
            Complex::new(real.exp(/**/), 0f32)
        }));
    
        let gy = DVector::from_iterator(tlb.h, (0..tlb.h).map(|y| {
            let real = -(y as f32 - tlb.ycenter).powi(2) / 2f32;
            Complex::new(real.exp(/**/), 0f32)
        }));
    
        let mut gaussian = &gy * &gx;
        let mat_template = DMatrix::zeros(tlb.h, tlb.w);
        let fwd_fft = planner.plan_fft_forward(tlb.length);
        let inv_fft = planner.plan_fft_inverse(tlb.length);
        let gaussian_slice = gaussian.as_mut_slice(/**/);
        fwd_fft.process(gaussian_slice);

        
    
        Self {
            hann_window: &hann_col * &hann_row,
            h: mat_template.clone(/**/),
            a: mat_template.clone(/**/),
            b: mat_template,
            g: gaussian,
            fwd_fft,
            inv_fft,
            tlb,
        }
    }
}

fn sqrt(x: f32) -> f32 {
    let x_half = 0.5 * x;
    let mut i = x.to_bits();
    i = 0x5f3759df - (i >> 1);
    let y = f32::from_bits(i);
    y * (1.5 - x_half * y * y)
}

fn main() {
    let mut planner = FftPlanner::<f32>::new(/**/);
    let mut img = GrayImage::new(800, 800);
    for pixel in img.pixels_mut(/**/) {
        *pixel = Luma([128u8]);
    }

    let bbox = BoundingBox::new(80, 80, 640, 640);
    let _ = TrackerMOSSE::new(&mut planner, bbox);
    
    let start = Instant::now();
    let bbox = BoundingBox::new(80, 80, 640, 640);
    let _ = TrackerMOSSE::new(&mut planner, bbox);
    println!("new: {:?}", start.elapsed());
}