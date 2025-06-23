#![allow(dead_code)]

mod optimal_dft;
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::cmp::Ordering::Equal;
use std::sync::Arc;
use rand::Rng;

use pyo3::prelude::*;
use numpy::PyReadonlyArray2;

// [x, y, w, h, prob]
type PyAbsBox = [f32; 5];
type VecCpxF32 = Vec<Complex32>;
type FftF32 = dyn Fft<f32>;

type Unit = ();
const UNIT: Unit = ();
const EPS: f32 = 0.00001;
const WARPS: usize = 8;

const EPSCPX: Complex32 = 
    Complex32::new(EPS, 0f32);

//

struct GrayImageWrap {
    image: image::GrayImage,
    height: isize,
    width: isize,
}

impl GrayImageWrap {
    fn from_py(raw: PyReadonlyArray2<u8>) -> Self {
        let array = raw.as_array(/**/);
        let shape = array.shape(/**/);
        let h = shape[0] as u32;
        let w = shape[1] as u32;
        
        let buffer: Vec<u8> = array.iter(/**/).cloned(/**/).collect(/**/);
        let image = image::GrayImage::from_raw(w, h, buffer).unwrap(/**/);
        Self { image, height: h as isize, width: w as isize }
    }
}

//

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
        let to_pseudo_complex = |value: &f32| Complex32::new(value / peak_value, 0f32);
        let mut gaussian: VecCpxF32 = gaussian.iter(/**/).map(to_pseudo_complex).collect(/**/);
        fft2dd(&mut gaussian, &fft, 1f32, size);

        let me = Self { fft, ifft, gaussian, hann, size, area, cx };
        Arc::new(me)
    }
}

//

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

//

#[derive(Clone, Copy)]
struct Config {
    min_psr: f32,
    learn_rate: f32,
    warp_scale: f32,
    mov_avg_alpha: f32,
    mov_avg_decay: f32,
    max_square: usize,
}

struct Mosse {
    tracks: f32,
    detects: f32,
    pre: Arc<Precomputed>,
    alloc_patch: Vec<f32>,
    alloc_resp: VecCpxF32,
    bbox: PyAbsBox,
    h: VecCpxF32,
    a: VecCpxF32,
    b: VecCpxF32,
    mac: f32,
    cx: f32,
    cy: f32,
    id: i32,
}

impl Mosse {
    fn new(mac: f32, tracks: f32, detects: f32, conf: &Config, cache: &mut Cache, giw: &GrayImageWrap, bbox: PyAbsBox, id: i32) -> Self {
        // Our cache won't grow too much since we adjust an actual bounding box size to discrete, small, finite set of optimal values
        let (size, cx, cy) = square(bbox, conf);
        let pre = cache.get(size);

        let mut apass = vec![Complex32::ZERO; WARPS * pre.area];
        let mut bpass = vec![Complex32::ZERO; WARPS * pre.area];
        let apass_ptr_uint = apass.as_mut_ptr(/**/) as usize;
        let bpass_ptr_uint = bpass.as_mut_ptr(/**/) as usize;

        let mut alloc_warp = vec![0f32; WARPS * pre.area];
        let mut alloc_patch = vec![0f32; pre.area];
        crop(giw, &pre, &mut alloc_patch, cx, cy);

        alloc_warp.par_chunks_mut(pre.area).enumerate(/**/).for_each(|itarget| unsafe {
            // Use pointer arithmetic to avoid per-warp apass/bpass heap allocations
            let apass_ptr = apass_ptr_uint as *mut Complex32;
            let bpass_ptr = bpass_ptr_uint as *mut Complex32;
            let shift = itarget.0 * pre.area;

            warp(&alloc_patch, itarget.1, conf.warp_scale, &pre, size);
            let mut target = preprocess(itarget.1, &pre.hann, 0f32);
            fft2d(&mut target, &pre, false);

            for n in 0..pre.area {
                let conjugate = target[n].conj(/**/);
                *apass_ptr.add(shift + n) = pre.gaussian[n] * conjugate;
                *bpass_ptr.add(shift + n) = target[n] * conjugate;
            }
        });

        let mut a = vec![Complex32::ZERO; pre.area];
        let mut b = vec![Complex32::ZERO; pre.area];
        let mut h = vec![Complex32::ZERO; pre.area];

        for n in 0..pre.area {
            let mut asum = Complex32::ZERO;
            let mut bsum = Complex32::ZERO;

            for idx in 0..WARPS {
                let shift = idx * pre.area;
                asum += apass[shift + n];
                bsum += bpass[shift + n];
            }

            a[n] = asum;
            b[n] = bsum;
            bsum += EPSCPX;
            h[n] = asum / bsum;
        }

        let alloc_resp = vec![Complex32::ZERO; pre.area];
        Self { tracks, detects, pre, alloc_resp, alloc_patch, 
            bbox, h, a, b, mac, cx, cy, id }
    }

    #[inline]
    fn update_consume(mut self, giw: &GrayImageWrap, conf: &Config) -> Option<Self> {
        // FFT cropped patch into frequency domain to avoid convolution
        let current = self.extract_fft(giw);
    
        for i in 0..self.pre.area {
            // collect fast filter response in frequency domain
            self.alloc_resp[i] = current[i] * self.h[i];
        }

        // IFFT collected response back into space domain
        fft2d(&mut self.alloc_resp, &self.pre, true);
    
        let mut sum = 0f32;
        let mut sumsq = 0f32;
        let mut maxv = f32::MIN;
        let mut idx = 0;

        for (i, c) in self.alloc_resp.iter(/**/).enumerate(/**/) {
            // Compute peak-to-sidelobe ratio in space domain
            sumsq += c.re.powi(2);
            sum += c.re;

            if c.re > maxv {
                maxv = c.re;
                idx = i;
            }
        }

        let mean = sum / self.pre.area as f32;
        let var = sumsq / self.pre.area as f32 - mean.powi(2);
        let peak_to_sidelobe_ratio = (maxv - mean) / var.sqrt(/**/);
        if peak_to_sidelobe_ratio.max(EPS) < conf.min_psr {
            return None
        }
    
        let row = (idx / self.pre.size) as isize;
        let col = (idx % self.pre.size) as isize;

        let resp_at = |r: isize, c: isize| -> f32 {
            let r = r.clamp(0, self.pre.size as isize - 1) as usize;
            let c = c.clamp(0, self.pre.size as isize - 1) as usize;
            self.alloc_resp[r * self.pre.size + c].re
        };

        let c_val = resp_at(row, col);
        let l_val = resp_at(row, col - 1);
        let r_val = resp_at(row, col + 1);
        let t_val = resp_at(row - 1, col);
        let b_val = resp_at(row + 1, col);

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

        self.bbox[0] += dx_shift;
        self.bbox[1] += dy_shift;
        self.cx += dx_shift;
        self.cy += dy_shift;
        self.tracks += 1f32;

        let next = self.extract_fft(giw);

        for i in 0..self.pre.area {
            let conjugate = next[i].conj(/**/) * conf.learn_rate;
            let an = self.pre.gaussian[i] * conjugate;
            let bn = next[i] * conjugate;

            self.a[i] = self.a[i] * (1f32 - conf.learn_rate) + an;
            self.b[i] = self.b[i] * (1f32 - conf.learn_rate) + bn;
            self.h[i] = self.a[i] / (self.b[i] + EPSCPX);
        }

        Some(self)
    }

    #[inline]
    fn extract_fft(&mut self, giw: &GrayImageWrap) -> VecCpxF32 {
        crop(giw, &self.pre, &mut self.alloc_patch, self.cx, self.cy);
        let mut buf = preprocess(&mut self.alloc_patch, &self.pre.hann, 0f32);
        fft2d(&mut buf, &self.pre, false);
        buf
    }
}

//

struct MultiMosse {
    trackers: Vec<Mosse>,
    max_targets: usize,
    threshold2: f32,
    next_id: i32,
}

impl MultiMosse {
    fn init(&mut self, giw: GrayImageWrap, conf: &Config, detections: &[PyAbsBox], cache: &mut Cache) {
        // There exists some detector which periodically gives us a list of bounding boxes
        // Our job is to track underlying objects for as long as possible until next list
        let mut replacement = Vec::with_capacity(self.max_targets);
        let can_take_old = self.max_targets - detections.len(/**/);

        for &bbox in detections {
            let (_, cx, cy) = square(bbox, conf);
            let mut best: Option<(usize, f32)> = None;

            for (i, m) in self.trackers.iter(/**/).enumerate(/**/) {
                let dist2 = (m.cx - cx).powi(2) + (m.cy - cy).powi(2);
                if dist2 < self.threshold2 && best.map_or(true, |val| dist2 < val.1) {
                    // Filter best match by closest distance and by predefined threshold
                    let new_best = (i, dist2);
                    best = Some(new_best);
                }
            }

            if let Some(tuple) = best {
                let Mosse { tracks, detects, mac, id, .. } = self.trackers.swap_remove(tuple.0);
                let mov_avg_update = mac * (1.0 - conf.mov_avg_alpha) + bbox[4] * conf.mov_avg_alpha;
                let m = Mosse::new(mov_avg_update, tracks, detects + 1f32, conf, cache, &giw, bbox, id);
                replacement.push(m);
            } else {
                self.next_id += 1;
                // Just spawn and add a new tracker for this brand new detection
                let m = Mosse::new(bbox[4], 1f32, 1f32, conf, cache, &giw, bbox, self.next_id);
                replacement.push(m);
            }
        }

        // We have removed all replaced trackers from current vector
        // here we retain remaining trackers who have survived an update
        self.trackers = self.trackers.drain(..).filter_map(|mut m| {
            // Gradually decrease moving average confidence
            m.mac *= conf.mov_avg_decay;
            m.update_consume(&giw, conf)
        }).collect(/**/);

        if can_take_old > 0 {
            self.trackers.sort_by(|track1, track2| {
                let qa = track1.detects / track1.tracks;
                let qb = track2.detects / track2.tracks;
                qb.partial_cmp(&qa).unwrap_or(Equal)
            });
            
            let take_old = self.trackers.len(/**/).min(can_take_old);
            let actual_taken_old = self.trackers.drain(..take_old);
            replacement.extend(actual_taken_old);
        }

        self.trackers = replacement;
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
    buf.par_chunks_mut(n).for_each(|row| fourier.process(row));
    let ptr_addr = buf.as_mut_ptr(/**/) as usize;

    (0..n).into_par_iter(/**/).for_each(|column| {
        let ptr = ptr_addr as *mut Complex32;
        let mut col = Vec::with_capacity(n);
        
        unsafe {
            for i in 0..n {
                let index = i * n + column;
                let val = ptr.add(index);
                col.push(*val);
            }

            fourier.process(&mut col);

            for i in 0..n {
                let index = i * n + column;
                let val = col[i] * scale;
                *ptr.add(index) = val;
            }
        }
    })
}

#[inline]
fn crop(giw: &GrayImageWrap, pre: &Precomputed, out: &mut [f32], cx: f32, cy: f32) {
    // Bilinear interpolation + mirror-border reflection for out-of-border regions
    // This method expects an image center and then computes its top/left point
    let center_to_left_top = (pre.size as f32 - 1f32) / 2f32;

    out.chunks_mut(pre.size).enumerate(/**/).for_each(|row| {
        let yf = cy + (row.0 as f32 - center_to_left_top);
        let y0 = yf.floor(/**/) as isize;
        let dy = yf - y0 as f32;
        let wy0 = 1f32 - dy;

        let y00 = reflect(y0, giw.height);
        let y01 = reflect(y0 + 1, giw.height);
        
        for (j, cell) in row.1.iter_mut(/**/).enumerate(/**/) {
            let xf = cx + (j as f32 - center_to_left_top);
            let x0 = xf.floor(/**/) as isize;
            let dx = xf - x0 as f32;
            let wx0 = 1f32 - dx;
            
            let x00 = reflect(x0, giw.width);
            let x10 = reflect(x0 + 1, giw.width);

            let i00 = giw.image.get_pixel(x00 as u32, y00 as u32)[0] as f32 + 1f32;
            let i10 = giw.image.get_pixel(x10 as u32, y00 as u32)[0] as f32 + 1f32;
            let i01 = giw.image.get_pixel(x00 as u32, y01 as u32)[0] as f32 + 1f32;
            let i11 = giw.image.get_pixel(x10 as u32, y01 as u32)[0] as f32 + 1f32;
            *cell = i00 * wx0 * wy0 + i10 * dx * wy0 + i01 * wx0 * dy + i11 * dx * dy;
        }
    });
}

#[inline]
fn preprocess(buf: &mut [f32], hann: &[f32], def: f32) -> VecCpxF32 {
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

    let buf_iter = buf.iter(/**/);
    let hann_iter = hann.iter(/**/);

    buf_iter.zip(hann_iter).map(|(v, &w)| {
        // Log-normalize and apply Hann window
        let norm_hann = (*v - mean) / std * w;
        Complex32::new(norm_hann, def)
    }).collect(/**/)
}

#[inline]
fn warp(src: &[f32], target: &mut [f32], scale: f32, pre: &Precomputed, n: usize) {
    // Applies random rotation + shear + scale to a given patch
    let mut rng = rand::rng(/**/);
    
    let ang = rng.random_range(-scale..scale);
    let c = ang.cos(/**/);
    let s = ang.sin(/**/);

    let w00 = c + rng.random_range(-scale..scale);
    let w01 = -s + rng.random_range(-scale..scale);
    let w10 = s + rng.random_range(-scale..scale);
    let w11 = c + rng.random_range(-scale..scale);

    let w02 = pre.cx - (w00 * pre.cx + w01 * pre.cx);
    let w12 = pre.cx - (w10 * pre.cx + w11 * pre.cx);

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
            let t0 = i00 * (1f32 - dx) + i10 * dx;
            let t1 = i01 * (1f32 - dx) + i11 * dx;
            let index = (i as usize) * n + (j as usize);
            target[index] = t0 * (1f32 - dy) + t1 * dy;
        }
    };
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

#[inline]
fn square(bbox: PyAbsBox, conf: &Config) -> (usize, f32, f32) {
    let [left, top, width, height, _] = bbox;
    let natural = width.max(height) as usize;
    let size = natural.min(conf.max_square);
    let size = optimal_dft::get_size(size);
    let cx = left + width / 2f32;
    let cy = top + height / 2f32;
    (size, cx, cy)
}

// PYTHON

#[pyclass]
struct PyTrack {
    #[pyo3(get)] left: i32,
    #[pyo3(get)] top: i32,
    #[pyo3(get)] width: i32,
    #[pyo3(get)] height: i32,
    #[pyo3(get)] prob: i32,
    #[pyo3(get)] trcs: i32,
    #[pyo3(get)] dts: i32,
    #[pyo3(get)] mac: i32,
    #[pyo3(get)] id: i32,
}

#[inline]
fn to_py_track(mosse: &Mosse) -> PyTrack {
    PyTrack { 
        left: mosse.bbox[0] as i32, 
        top: mosse.bbox[1] as i32,
        width: mosse.bbox[2] as i32, 
        height: mosse.bbox[3] as i32, 
        prob: mosse.bbox[4] as i32, 
        trcs: mosse.tracks as i32, 
        dts: mosse.detects as i32, 
        mac: mosse.mac as i32, 
        id: mosse.id 
    }
}

#[pyclass]
struct PyMosse {
    mm: MultiMosse,
    cache: Cache,
    conf: Config,
}

#[pymethods]
impl PyMosse {
    fn init(&mut self, detections: Vec<PyAbsBox>, gray_image: PyReadonlyArray2<u8>) -> PyResult<Vec<PyTrack>> {
        self.mm.init(GrayImageWrap::from_py(gray_image), &self.conf, &detections, &mut self.cache);
        let out = self.mm.trackers.iter(/**/).map(to_py_track).collect(/**/);
        Ok(out)
    }

    #[new]
    fn new(max_targets: usize, threshold: f32, min_psr: f32, learn_rate: f32, warp_scale: f32, max_square: usize) -> Self {
        let mm = MultiMosse { trackers: Vec::new(/**/), max_targets, threshold2: threshold * threshold, next_id: 0 };
        let conf = Config { min_psr, learn_rate, warp_scale, mov_avg_alpha: 0.2, mov_avg_decay: 0.995, max_square };
        let cache = Cache::new(optimal_dft::LEN);
        Self { mm, cache, conf }
    }
}

#[pymodule]
fn rmosse(m: &Bound<'_, PyModule>) -> PyResult<Unit> {
    m.add_class::<PyMosse>(/**/)?;
    m.add_class::<PyTrack>(/**/)?;
    Ok(UNIT)
}