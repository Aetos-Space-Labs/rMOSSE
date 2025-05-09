#![allow(dead_code)]

mod optimal_dft;
use rayon::prelude::*;
use std::sync::atomic::{AtomicPtr, Ordering};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use image::GrayImage;
use std::sync::Arc;
use rand::Rng;

type VecCpxF32 = Vec<Complex32>;
type FftF32 = dyn Fft<f32>;

const PSR: f32 = 5.7;
const WARP: f32 = 0.2;
const EPS: f32 = 0.00001;
const LEARNRATE: f32 = 0.2;
const MAXTARGETS: usize = 10;

const EPSCPX: Complex32 = Complex32::new(EPS, 0f32);

#[derive(Clone, Copy, Debug)]
struct AbsBox {
    left: f32,
    top: f32,
    width: f32,
    height: f32,
}

impl AbsBox {
    fn square(&self) -> (usize, f32, f32) {
        let size = self.width.max(self.height) as usize;
        let size = optimal_dft::get_optimal_dft_size(size);
        let cx = self.left + self.width / 2f32;
        let cy = self.top + self.height / 2f32;
        (size, cx, cy)
    }
}

struct Precomputed {
    fft: Arc<FftF32>,
    ifft: Arc<FftF32>,
    gaussian: VecCpxF32,
    hann: Vec<f32>,
    size: usize,
    area: usize,
    cx: f32,
}

impl Precomputed {
    fn new(planner: &mut FftPlanner<f32>, size: usize) -> Arc<Self> {
        let nm = (size - 1) as f32;
        let area = size * size;
        let cx = nm / 2f32;
        let cy = nm / 2f32;
        
        // Hann window tapering function
        // Combats Heisenberg uncertainty
        let mut hann = Vec::with_capacity(area);
        
        // Represents an ideal response
        // Peaks at center of bounding box
        let mut gaussian = Vec::with_capacity(area);
        let ifft = planner.plan_fft_inverse(size);
        let fft = planner.plan_fft_forward(size);
        let pi2 = 2f32 * std::f32::consts::PI;
        let sigma22 = 2f32 * 2f32 * 2f32;

        for i in 0..size {
            // Optimization: compute both gaussian and hann here
            let wy = 1f32 - (pi2 * i as f32 / nm).cos(/**/);
            let dy = i as f32 - cy;

            for j in 0..size {
                let dx = j as f32 - cx;
                let val = (-(dx * dx + dy * dy) / sigma22).exp(/**/);
                let wx = 1f32 - (pi2 * j as f32 / nm).cos(/**/);
                hann.push(wy * wx / 4f32);
                gaussian.push(val);
            }
        }

        let peak_value = gaussian.iter(/**/).cloned(/**/).fold(0f32, f32::max);
        for current_value in &mut gaussian { *current_value /= peak_value; }
        let mut gaussian = to_complex(&gaussian, 0f32);
        fft2dd(&mut gaussian, &fft, 1f32, size);

        let me = Self { fft, ifft, gaussian, hann, size, area, cx };
        Arc::new(me)
    }
}

struct Cache {
    planner: FftPlanner<f32>,
    cache: Vec<Arc<Precomputed>>,
}

impl Cache {
    fn new(len: usize) -> Self {
        let planner = FftPlanner::new(/**/);
        let cache = Vec::with_capacity(len);
        Self { planner, cache }
    }

    fn get(&mut self, size: usize) -> Arc<Precomputed> {
        if let Some(idx) = self.cache.iter(/**/).position(|pre| pre.size == size) {
            // Internally we only do square bounding boxes to simplify things
            // Among other perks this allows for simple caching mechanism
            return self.cache[idx].clone(/**/);
        }

        let pre_arc = Precomputed::new(&mut self.planner, size);
        let pre_arc_clone = pre_arc.clone(/**/);
        self.cache.push(pre_arc_clone);
        pre_arc
    }
}

struct Mosse {
    pre: Arc<Precomputed>,
    scratch: VecCpxF32,
    h: VecCpxF32,
    a: VecCpxF32,
    b: VecCpxF32,
    bbox: AbsBox,
    psr: f32,
    cx: f32,
    cy: f32,
}

impl Mosse {
    fn new(cache: &mut Cache, img: &GrayImage, bbox: AbsBox) -> Self {
        // Since we adjust an actual bbox size our cache won't grow too much
        let (size, cx, cy) = bbox.square(/**/);
        let pre = cache.get(size);

        let patch = crop(img, &pre, cx, cy);
        let parts: Vec<(VecCpxF32, VecCpxF32)> =
            (0..8).into_par_iter(/**/).map(|_| {
                let mut target = vec![0f32; pre.area];
                warp(&patch, &mut target, pre.cx, pre.cx, size);
                preprocess(&mut target, &pre.hann, 0f32);

                let mut target = to_complex(&target, 0f32);
                fft2d(&mut target, &pre, false);

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

        let h = aiter.zip(biter).map(|ab| {
            *ab.0 / (*ab.1 + EPSCPX)
        }).collect(/**/);
        
        let psr = std::f32::MAX;
        let scratch = vec![Complex32::ZERO; pre.area];
        Self { pre, scratch, h, a, b, bbox, psr, cx, cy }
    }

    fn update(&mut self, img: &GrayImage) -> Option<AbsBox> {
        let current = self.extract_and_fft(img);
    
        for i in 0..self.pre.area {
            self.scratch[i] = current[i] * self.h[i];
        }

        fft2d(&mut self.scratch, &self.pre, true);
    
        let mut sum = 0f32;
        let mut sumsq = 0f32;
        let mut maxv = f32::MIN;
        let mut idx = 0;

        for (i, c) in self.scratch.iter(/**/).enumerate(/**/) {
            // Compute peak-to-sidelobe ratio in space domain
            sumsq += c.re * c.re;
            sum += c.re;

            if c.re > maxv {
                maxv = c.re;
                idx = i;
            }
        }

        let mean = sum / self.pre.area as f32;
        let var = sumsq / self.pre.area as f32 - mean * mean;
        self.psr = (maxv - mean) / var.sqrt(/**/).max(EPS);

        if self.psr < PSR {
            return None;
        }
    
        let row = idx / self.pre.size;
        let col = idx % self.pre.size;

        let resp_at = |r: isize, c: isize| -> f32 {
            let r = r.clamp(0, self.pre.size as isize - 1) as usize;
            let c = c.clamp(0, self.pre.size as isize - 1) as usize;
            self.scratch[r * self.pre.size + c].re
        };

        let c_val = resp_at(row as isize, col as isize);
        let l_val = resp_at(row as isize, col as isize - 1);
        let r_val = resp_at(row as isize, col as isize + 1);
        let t_val = resp_at(row as isize - 1, col as isize);
        let b_val = resp_at(row as isize + 1, col as isize);

        let denom_x = 2f32 * c_val - l_val - r_val;
        let denom_y = 2f32 * c_val - t_val - b_val;

        let dx = if denom_x.abs(/**/) < EPS { 0f32 } else { 
            (r_val - l_val) / (2f32 * denom_x) 
        };
        
        let dy = if denom_y.abs(/**/) < EPS { 0f32 } else { 
            (b_val - t_val) / (2f32 * denom_y) 
        };

        let offset = (self.pre.size as f32 - 1f32) / 2f32;
        let dx_shift = col as f32 + dx - offset;
        let dy_shift = row as f32 + dy - offset;

        self.bbox.left += dx_shift;
        self.bbox.top += dy_shift;
        self.cx += dx_shift;
        self.cy += dy_shift;

        let next = self.extract_and_fft(img);

        for i in 0..self.pre.area {
            let an = self.pre.gaussian[i] * next[i].conj(/**/) * LEARNRATE;
            let bn = next[i] * next[i].conj(/**/) * LEARNRATE;
            self.a[i] = self.a[i] * (1f32 - LEARNRATE) + an;
            self.b[i] = self.b[i] * (1f32 - LEARNRATE) + bn;
            self.h[i] = self.a[i] / (self.b[i] + EPSCPX);
        }
        
        Some(self.bbox)
    }

    #[inline]
    fn extract_and_fft(&self, img: &GrayImage) -> VecCpxF32 {
        let mut buf = crop(img, &self.pre, self.cx, self.cy);
        preprocess(&mut buf, &self.pre.hann, 0f32);
        let mut buf = to_complex(&buf, 0f32);
        fft2d(&mut buf, &self.pre, false);
        buf
    }
}

struct MultiMosse {
    trackers: Vec<Mosse>,
    threshold2: f32,
}

impl MultiMosse {
    fn step(&mut self, detections: &[AbsBox], img: &GrayImage, cache: &mut Cache) -> Vec<AbsBox> {
        // There exists some detector which periodically gives us a list of found bounding boxes
        // Our job is to track underlying objects for as long as possible until next list
        let mut new_trackers = Vec::with_capacity(MAXTARGETS);
        let mut survivors = Vec::with_capacity(MAXTARGETS);
        let mut replaced = vec![false; MAXTARGETS];

        for &bbox in detections {
            let (_, cx, cy) = bbox.square(/**/);
            let mut best: Option<(usize, f32)> = None;
            for (i, m) in self.trackers.iter(/**/).enumerate(/**/) {
                // Unless already replaced, find the closest active tracker
                // That tracker then gets replaced, we treat it as best candidate

                if !replaced[i] {
                    let dx = m.cx - cx;
                    let dy = m.cy - cy;
                    let dist2 = dx * dx + dy * dy;

                    if dist2 < self.threshold2 && best.map_or(true, |val| dist2 < val.1) {
                        // Filter by closest distance and by predefined threshold
                        let new_best = (i, dist2);
                        best = Some(new_best);
                    }
                }
            }

            if let Some(val) = best {
                let mosse = Mosse::new(cache, img, bbox);
                new_trackers.push(mosse);
                replaced[val.0] = true;
            } else {
                let num_replaced = replaced.iter(/**/).filter(|&&value| value).count(/**/);
                if self.trackers.len(/**/) - num_replaced + new_trackers.len(/**/) < MAXTARGETS {
                    // We never replace active trackers and let them naturally die out instead
                    let mosse = Mosse::new(cache, img, bbox);
                    new_trackers.push(mosse);
                }
            }
        }

        for (i, mut m) in self.trackers.drain(..).enumerate(/**/) {
            // Retain trackers which are not replaced and have high PSR
            // We do update on untouched and replace with new in one pass
            if !replaced[i] && m.update(img).is_some(/**/) { 
                survivors.push(m);
            }
        }

        survivors.extend(new_trackers);
        survivors.truncate(MAXTARGETS);
        self.trackers = survivors;

        self.trackers
            .iter(/**/)
            .map(|m| m.bbox)
            .collect(/**/)
    }
}

// Utils

#[inline]
fn fft2d(buf: &mut [Complex32], pre: &Precomputed, ifft: bool) {
    let scale = if ifft { 1f32 / pre.area as f32 } else { 1f32 };
    let fourier = if ifft { &pre.ifft } else { &pre.fft };
    fft2dd(buf, fourier, scale, pre.size)
}

#[inline]
fn fft2dd(buf: &mut [Complex32], fourier: &Arc<FftF32>, scale: f32, n: usize) {
    // Applies 2D (row-wise, then column-wise) FFT/IFFT, avoids full transpose
    buf.par_chunks_mut(n).for_each(|value| {
        fourier.process(value);
    });

    let ptr = buf.as_mut_ptr(/**/);
    let ptr_atom = AtomicPtr::new(ptr);

    (0..n).into_par_iter(/**/).for_each(|row| {
        let ptr = ptr_atom.load(Ordering::Relaxed);
        let mut col = Vec::with_capacity(n);
        
        unsafe {
            for i in 0..n {
                let index = i * n + row;
                let val = ptr.add(index);
                col.push(*val);
            }

            fourier.process(&mut col);

            for i in 0..n {
                let index = i * n + row;
                let val = col[i] * scale;
                *ptr.add(index) = val;
            }
        }
    })
}

#[inline]
fn crop(img: &GrayImage, pre: &Precomputed, cx: f32, cy: f32) -> Vec<f32> {
    // Bilinear interpolation + mirror-border reflection for out-of-border regions
    // This method expects an image center and then computes its top/left point
    let center_to_left_top = (pre.size as f32 - 1f32) / 2f32;
    let mut out = vec![0f32; pre.area];
    let h = img.height(/**/) as isize;
    let w = img.width(/**/) as isize;

    out.par_chunks_mut(pre.size).enumerate(/**/).for_each(|row| {
           let yf = cy + (row.0 as f32 - center_to_left_top);
           let y0 = yf.floor(/**/) as isize;
           let dy = yf - y0 as f32;
           let wy0 = 1f32 - dy;

           let y00 = reflect(y0, h);
           let y01 = reflect(y0 + 1, h);
           
           for (j, cell) in row.1.iter_mut(/**/).enumerate(/**/) {
                let xf = cx + (j as f32 - center_to_left_top);
                let x0 = xf.floor(/**/) as isize;
                let dx = xf - x0 as f32;
                let wx0 = 1f32 - dx;
                
                let x00 = reflect(x0, w);
                let x10 = reflect(x0 + 1, w);

                let i00 = img.get_pixel(x00 as u32, y00 as u32)[0] as f32 + 1f32;
                let i10 = img.get_pixel(x10 as u32, y00 as u32)[0] as f32 + 1f32;
                let i01 = img.get_pixel(x00 as u32, y01 as u32)[0] as f32 + 1f32;
                let i11 = img.get_pixel(x10 as u32, y01 as u32)[0] as f32 + 1f32;
                *cell = i00 * wx0 * wy0 + i10 * dx * wy0 + i01 * wx0 * dy + i11 * dx * dy;
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
fn to_complex(rc: &[f32], im: f32) -> VecCpxF32 {
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
    let mut result = idx;

    while result < 0 || result >= len {
        if result < 0 { result = -result; } 
        else { result = 2 * len - 2 - result; }
    }
    
    result as usize
}

fn main(/**/) {
    let img = GrayImage::new(800u32, 800u32);
    let bbox = AbsBox { left: 80.0, top: 80.0, width: 640.0, height: 640.0 };
    let mut cache = Cache::new(optimal_dft::LEN);
    let _ = Mosse::new(&mut cache, &img, bbox);

    let t0 = std::time::Instant::now(/**/);
    let mut mosse = Mosse::new(&mut cache, &img, bbox);
    let init_time = t0.elapsed(/**/);

    let t1 = std::time::Instant::now(/**/);
    let _ = mosse.update(&img);
    let upd_time = t1.elapsed(/**/);

    println!("init: {:.3?} ms", init_time);
    println!("update: {:.3?} ms", upd_time);
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use image::{ImageReader, Rgb};
    use super::*;

    #[test]
    fn multi_mosse_tracking(/**/) -> Result<(/**/), Box<dyn Error>> {
        let gray1 = ImageReader::open("frame1.png")?.decode(/**/)?.to_luma8(/**/);
        let gray2 = ImageReader::open("frame2.png")?.decode(/**/)?.to_luma8(/**/);
        let mut disp = ImageReader::open("frame2.png")?.decode(/**/)?.to_rgb8(/**/);
        let mut cache = Cache::new(optimal_dft::LEN);

        let init_boxes = [
            AbsBox { left: 195f32, top: 150f32, width: 80f32, height: 50f32 },
            AbsBox { left: 600f32, top: 200f32, width: 70f32, height: 50f32 },
            AbsBox { left: 720f32, top: 220f32, width: 80f32, height: 60f32 },
        ];

        // Initialize trackers on first frame, do not update them
        let trackers: Vec<Mosse> = init_boxes.iter(/**/).map(|&bbox| {
            Mosse::new(&mut cache, &gray1, bbox)
        }).collect(/**/);

        let mut multi = MultiMosse { trackers, threshold2: 2500f32 };
        // Pretend we have not detected anything, trackers are on their own
        let out = multi.step(&[], &gray2, &mut cache);

        for b in &out {
            let x0 = b.left as u32;
            let y0 = b.top as u32;

            let x1 = x0 + b.width as u32;
            let y1 = y0 + b.height as u32;
            let rgb = Rgb([255, 0, 0]);

            for x in x0..x1 {
                disp.put_pixel(x, y0, rgb);
                disp.put_pixel(x, y1, rgb);
            }

            for y in y0..y1 {
                disp.put_pixel(x0, y, rgb);
                disp.put_pixel(x1, y, rgb);
            }
        }

        disp.save("result.png")?;
        let nothing_lost = out.len(/**/) == init_boxes.len(/**/);
        assert!(nothing_lost, "lost trackers");
        Ok((/**/))
    }
}