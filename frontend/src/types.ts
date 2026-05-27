/**
 * TypeScript interfaces mirroring the FastAPI AnalysisResponse schema.
 * Keep in sync with backend/models.py.
 */

export interface FrameResult {
  frame_index: number;
  timestamp_sec: number;
  prob_fake: number;
  ear_score: number;
  artifact_score: number;
  fft_score: number;
  laplacian_score: number;
  face_detected: boolean;
  face_crop_b64: string | null;
}

export interface AnalysisResponse {
  // Identity
  video_name: string;

  // Top-level verdict
  verdict: 'FAKE' | 'REAL' | 'UNCERTAIN';
  confidence: number;
  prob_fake_mean: number;

  // Signal breakdown
  quality_weighted_prob_fake: number;
  temporal_score: number;
  rppg_fake_score: number;
  rppg_available: boolean;

  // Model metadata
  model_used: 'ensemble_learned' | 'equal_weights';
  cnn_active: boolean;

  // Registry provenance (v2)
  model_id:   string;
  model_type: string;
  model_f1:   number | null;

  // Frame counts + video metadata
  frames_analyzed: number;
  frames_sampled: number;
  fps: number;
  duration_sec: number;

  // Per-frame data
  frames: FrameResult[];

  // Warnings
  warnings: string[];
}
