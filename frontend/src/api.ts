/**
 * Typed axios wrapper for the deepfake detector API.
 *
 * The Vite dev server proxies /api/* → http://localhost:8000/*
 * so we call /api/analyze here and the backend receives /analyze.
 */

import axios from 'axios';
import type { AnalysisResponse } from './types';

/**
 * POST /api/analyze
 *
 * @param file     - Video file selected by the user
 * @param nFrames  - Number of frames to sample (1–60)
 * @returns        - Parsed AnalysisResponse
 */
export async function postAnalyze(
  file: File,
  nFrames: number = 12,
): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append('video', file);
  form.append('n_frames', String(nFrames));

  const { data } = await axios.post<AnalysisResponse>('/api/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120_000, // 2-minute timeout for large videos
  });

  return data;
}
