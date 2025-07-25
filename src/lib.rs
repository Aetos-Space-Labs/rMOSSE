mod optimal_dft;
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::cmp::Ordering::Equal;
use std::f32::consts::TAU;
use std::sync::Arc;
use rand::Rng;

use pyo3::prelude::*;
use numpy::PyReadonlyArray2;
use numpy::ndarray::{ArrayView, Ix2};

// [x, y, w, h, prob]
type PyAbsBox = [f32; 5];
type VecCpxF32 = Vec<Complex32>;
type ImgView<'a> = ArrayView<'a, u8, Ix2>;
type FftF32 = dyn Fft<f32>;

type Unit = ();
const UNIT: Unit = ();
const EPS: f32 = 0.00001;
const WARPS: usize = 6;

const EPSCPX: Complex32 = 
    Complex32::new(EPS, 0f32);

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
        let area = size.pow(2);
        let cx = nm / 2f32;
        
        // Hann window tapering function
        // Combats Heisenberg uncertainty
        let mut hann = Vec::with_capacity(area);
        
        // Represents an ideal response
        // Peaks at center of bounding box
        let mut gaussian = Vec::with_capacity(area);
        let ifft = planner.plan_fft_inverse(size);
        let fft = planner.plan_fft_forward(size);
        let sigma22 = 2f32.powi(3);

        for i in 0..size {
            // Optimization: compute both gaussian and hann here
            let wy = 1f32 - (TAU * i as f32 / nm).cos(/**/);
            let dy = i as f32 - cx;

            for j in 0..size {
                let dx = j as f32 - cx;
                let val = (-(dx * dx + dy * dy) / sigma22).exp(/**/);
                let wx = 1f32 - (TAU * j as f32 / nm).cos(/**/);
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

    #[inline]
    fn preprocess(&self, buf: &mut [f32], out: &mut [Complex32], base: f32) {
        // Log-normalize and apply Hann window
        let len = buf.len(/**/);
        let mut sum = base;
        let mut sq = base;

        for value in buf.iter_mut(/**/) {
            *value = value.ln(/**/);
            sq += value.powi(2);
            sum += *value;
        }

        let mean = sum / len as f32;
        let variance = sq / len as f32 - mean.powi(2);
        let std = variance.sqrt(/**/) + EPS;

        for i in 0..len {
            let norm_hann = (buf[i] - mean) / std * self.hann[i];
            out[i] = Complex32::new(norm_hann, base);
        }
    }

    #[inline]
    fn crop(&self, iw: &ImgView, out: &mut [f32], cx: f32, cy: f32) {
        // Bilinear interpolation + mirror-border reflection for out-of-border regions
        // This method expects an image center and then computes its top/left point
        let center_to_left_top = (self.size as f32 - 1f32) / 2f32;
        let height = iw.nrows(/**/) as isize;
        let width = iw.ncols(/**/) as isize;

        out.chunks_mut(self.size).enumerate(/**/).for_each(|row| {
            let yf = cy + (row.0 as f32 - center_to_left_top);
            let y0 = yf.floor(/**/) as isize;
            let dy = yf - y0 as f32;
            let wy0 = 1f32 - dy;

            let y00 = reflect(y0, height);
            let y01 = reflect(y0 + 1, height);
            
            for (j, cell) in row.1.iter_mut(/**/).enumerate(/**/) {
                let xf = cx + (j as f32 - center_to_left_top);
                let x0 = xf.floor(/**/) as isize;
                let dx = xf - x0 as f32;
                let wx0 = 1f32 - dx;
                
                let x00 = reflect(x0, width);
                let x10 = reflect(x0 + 1, width);

                let i00 = iw[(y00, x00)] as f32 + 1f32;
                let i10 = iw[(y00, x10)] as f32 + 1f32;
                let i01 = iw[(y01, x00)] as f32 + 1f32;
                let i11 = iw[(y01, x10)] as f32 + 1f32;
                *cell = i00 * wx0 * wy0 + i10 * dx * wy0 + i01 * wx0 * dy + i11 * dx * dy;
            }
        });
    }

    #[inline]
    fn warp(&self, src: &[f32], target: &mut [f32], scale: f32, n: usize) {
        // Applies random rotation + shear + scale to a given patch
        let mut rng = rand::rng(/**/);
        
        let ang = rng.random_range(-scale..scale);
        let c = ang.cos(/**/);
        let s = ang.sin(/**/);

        let w00 = c + rng.random_range(-scale..scale);
        let w01 = -s + rng.random_range(-scale..scale);
        let w10 = s + rng.random_range(-scale..scale);
        let w11 = c + rng.random_range(-scale..scale);

        let w02 = self.cx - (w00 * self.cx + w01 * self.cx);
        let w12 = self.cx - (w10 * self.cx + w11 * self.cx);

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
        }
    }
}

//

struct Cache {
    planner: FftPlanner<f32>,
    cache: Vec<Arc<Precomputed>>,
}

impl Cache {
    fn new(len: usize) -> Self {
        let cache = Vec::with_capacity(len);
        Self { planner: FftPlanner::new(/**/), cache }
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

struct Config {
    mov_avg_alpha: f32,
    mov_avg_decay: f32,
    max_square: usize,
    warp_scale: f32,
    learn_rate: f32,
    min_psr: f32,
}

struct Mosse {
    tracks: f32,
    detects: f32,
    pre: Arc<Precomputed>,
    alloc_patch_space: Vec<f32>,
    alloc_patch_freq: VecCpxF32,
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
    fn new(mac: f32, tracks: f32, detects: f32, conf: &Config, cache: &mut Cache, iw: &ImgView, bbox: PyAbsBox, id: i32) -> Self {
        // Our cache won't grow too much since we adjust an actual bounding box size to discrete, small, finite set of optimal values
        let (size, cx, cy) = square(bbox, conf);
        let pre = cache.get(size);

        let mut total_warps = vec![0f32; WARPS * pre.area];
        let mut alloc_patch_space = vec![0f32; pre.area];
        pre.crop(iw, &mut alloc_patch_space, cx, cy);

        let parts: Vec<(VecCpxF32, VecCpxF32)> = 
            total_warps.par_chunks_mut(pre.area).map(|warp| {
                let mut target_freq = vec![Complex32::ZERO; pre.area];
                pre.warp(&alloc_patch_space, warp, conf.warp_scale, size);
                pre.preprocess(warp, &mut target_freq, 0f32);
                fft2d(&mut target_freq, &pre, false);

                let mut apass = Vec::with_capacity(pre.area);
                let mut bpass = Vec::with_capacity(pre.area);

                for idx in 0..pre.area {
                    let conjugate = target_freq[idx].conj(/**/);
                    apass.push(pre.gaussian[idx] * conjugate);
                    bpass.push(target_freq[idx] * conjugate);
                }

                (apass, bpass)
            }).collect(/**/);

        let mut a = Vec::with_capacity(pre.area);
        let mut b = Vec::with_capacity(pre.area);
        let mut h = Vec::with_capacity(pre.area);

        for i in 0..pre.area {
            let mut asum = Complex32::ZERO;
            let mut bsum = Complex32::ZERO;

            for (an, bn) in &parts {
                asum += an[i];
                bsum += bn[i];
            }

            a.push(asum);
            b.push(bsum);
            bsum += EPSCPX;
            h.push(asum / bsum);
        }

        let alloc_resp = vec![Complex32::ZERO; pre.area];
        let alloc_patch_freq = vec![Complex32::ZERO; pre.area];
        Self { tracks, detects, pre, alloc_patch_space, alloc_patch_freq, 
            alloc_resp, bbox, h, a, b, mac, cx, cy, id }
    }

    #[inline]
    fn update_consume(mut self, iw: &ImgView, conf: &Config) -> Option<Self> {
        // FFT cropped patch into frequency domain to avoid convolution
        self.extract_fft(iw);
    
        for i in 0..self.pre.area {
            // collect fast filter response in frequency domain
            self.alloc_resp[i] = self.alloc_patch_freq[i] * self.h[i];
        }

        // IFFT collected response back into space domain
        fft2d(&mut self.alloc_resp, &self.pre, true);
    
        let mut sum = 0f32;
        let mut sumsq = 0f32;
        let mut maxv = f32::MIN;
        let mut idx = 0;

        for (i, c) in self.alloc_resp.iter(/**/).enumerate(/**/) {
            // Compute peak-to-sidelobe ratio in space domain
            // Peak is single pixel with max filter response
            sumsq += c.re.powi(2);
            sum += c.re;

            if c.re > maxv {
                maxv = c.re;
                idx = i;
            }
        }

        let mean = sum / self.pre.area as f32;
        let var = sumsq / self.pre.area as f32 - mean.powi(2);
        let psr = (maxv - mean) / var.max(EPS).sqrt(/**/);

        if psr.max(EPS) < conf.min_psr {
            return None
        }
    
        let peak_row = idx / self.pre.size;
        let peak_col = idx % self.pre.size;

        let resp_at = |r: usize, c: usize| -> f32 {
            let r = r.clamp(0, self.pre.size - 1);
            let c = c.clamp(0, self.pre.size - 1);
            let location = r * self.pre.size + c;
            self.alloc_resp[location].re
        };

        let c_val = resp_at(peak_row, peak_col);
        let l_val = resp_at(peak_row, peak_col - 1);
        let r_val = resp_at(peak_row, peak_col + 1);
        let t_val = resp_at(peak_row - 1, peak_col);
        let b_val = resp_at(peak_row + 1, peak_col);

        let denom_x = 2f32 * c_val - l_val - r_val;
        let denom_y = 2f32 * c_val - t_val - b_val;

        let dx = if denom_x.abs(/**/) < EPS { 0f32 } else { 
            (r_val - l_val) / (2f32 * denom_x) 
        };
        
        let dy = if denom_y.abs(/**/) < EPS { 0f32 } else { 
            (b_val - t_val) / (2f32 * denom_y) 
        };

        let offset = (self.pre.size as f32 - 1f32) / 2f32;
        let dx_shift = peak_col as f32 + dx - offset;
        let dy_shift = peak_row as f32 + dy - offset;

        self.bbox[0] += dx_shift;
        self.bbox[1] += dy_shift;
        self.cx += dx_shift;
        self.cy += dy_shift;
        self.tracks += 1f32;
        Some(self)
    }

    #[inline]
    fn learn(&mut self, iw: &ImgView, conf: &Config) {
        // We explicitly separate learning phase from update phase here
        // This way API user can obtain result faster and learn afterwards
        self.extract_fft(iw);

        for i in 0..self.pre.area {
            let lr = 1f32 - conf.learn_rate;
            let frequency_slot = self.alloc_patch_freq[i];
            let conjugate = frequency_slot.conj(/**/) * conf.learn_rate;
            self.a[i] = self.a[i] * lr + self.pre.gaussian[i] * conjugate;
            self.b[i] = self.b[i] * lr + frequency_slot * conjugate;
            self.h[i] = self.a[i] / (self.b[i] + EPSCPX);
        }
    }

    #[inline]
    fn extract_fft(&mut self, iw: &ImgView) {
        self.pre.crop(iw, &mut self.alloc_patch_space, self.cx, self.cy);
        self.pre.preprocess(&mut self.alloc_patch_space, &mut self.alloc_patch_freq, 0f32);
        fft2d(&mut self.alloc_patch_freq, &self.pre, false);
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
    fn init(&mut self, iw: ImgView, conf: &Config, detections: &[PyAbsBox], cache: &mut Cache) {
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
                let m = Mosse::new(mov_avg_update, tracks, detects + 1f32, conf, cache, &iw, bbox, id);
                replacement.push(m);
            } else {
                self.next_id += 1;
                // Just spawn and add a new tracker for this brand new detection
                let m = Mosse::new(bbox[4], 1f32, 1f32, conf, cache, &iw, bbox, self.next_id);
                replacement.push(m);
            }
        }

        // We have removed all replaced trackers from current vector
        // here we retain remaining trackers who have survived an update
        self.trackers = self.trackers.drain(..).filter_map(|mut m| {
            // Gradually decrease moving average confidence
            m.mac *= conf.mov_avg_decay;
            m.update_consume(&iw, conf)
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
fn fft2dd(buf: &mut [Complex32], fourier: &Arc<FftF32>, scale: f32, size: usize) {
    // Applies 2D (row-wise, then column-wise) FFT/IFFT, avoids full transpose
    buf.par_chunks_mut(size).for_each(|row| {
        fourier.process(row);
    });

    let ptr_addr = buf.as_mut_ptr(/**/) as usize;
    (0..size).into_par_iter(/**/).for_each(|column| unsafe {
        let mut col = Vec::with_capacity(size);
        let ptr = ptr_addr as *mut Complex32;
        
        for i in 0..size {
            let index = i * size + column;
            let val = ptr.add(index);
            col.push(*val);
        }

        fourier.process(&mut col);

        for i in 0..size {
            let index = i * size + column;
            let val = col[i] * scale;
            *ptr.add(index) = val;
        }
    })
}

#[inline]
fn reflect(idx: isize, len: isize) -> usize {
    // Mirror‐border reflect with 101 behavior
    let mut result = idx;

    while result < 0 || result >= len {
        if result < 0 { result = -result } 
        else { result = 2 * len - 2 - result }
    }
    
    result as usize
}

#[inline]
fn square(bbox: PyAbsBox, conf: &Config) -> (usize, f32, f32) {
    let [left, top, width, height, _] = bbox;

    let natural = width.max(height).ceil(/**/) as usize;
    let clamped = natural.clamp(optimal_dft::MIN, conf.max_square);
    (optimal_dft::get_size(clamped), left + width / 2f32, top + height / 2f32)
}

// PYTHON

#[pyclass]
struct PyTrack {
    #[pyo3(get)] bbox: PyAbsBox,
    #[pyo3(get)] trcs: i32,
    #[pyo3(get)] dts: i32,
    #[pyo3(get)] mac: i32,
    #[pyo3(get)] id: i32,
}

#[inline]
fn to_py_track(mosse: &Mosse) -> PyTrack {
    PyTrack { 
        bbox: mosse.bbox,
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
        self.mm.init(gray_image.as_array(/**/), &self.conf, &detections, &mut self.cache);
        let out = self.mm.trackers.iter(/**/).map(to_py_track).collect(/**/);
        Ok(out)
    }

    fn learn(&mut self, gray_image: PyReadonlyArray2<u8>) {
        let gray_image_view = gray_image.as_array(/**/);
        for tracker in self.mm.trackers.iter_mut(/**/) {
            tracker.learn(&gray_image_view, &self.conf);
        }
    }

    #[new]
    fn new(max_targets: usize, threshold: f32, min_psr: f32, learn_rate: f32, warp_scale: f32, max_square: usize) -> Self {
        let conf = Config { mov_avg_alpha: 0.2, mov_avg_decay: 0.995, max_square, warp_scale, learn_rate, min_psr };
        let mm = MultiMosse { trackers: Vec::new(/**/), max_targets, threshold2: threshold.powi(2), next_id: 0 };
        Self { mm, cache: Cache::new(optimal_dft::LEN), conf }
    }
}

#[pymodule]
fn rmosse(m: &Bound<'_, PyModule>) -> PyResult<Unit> {
    m.add_class::<PyMosse>(/**/)?;
    m.add_class::<PyTrack>(/**/)?;
    Ok(UNIT)
}
