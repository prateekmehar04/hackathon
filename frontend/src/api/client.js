/**
 * API client for the ECG Arrhythmia Classifier backend.
 */
import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

/** Fetch available demo ECG samples */
export async function fetchSamples() {
  const { data } = await api.get('/samples')
  return data
}

/** Load a specific demo sample by record ID */
export async function fetchSample(recordId) {
  const { data } = await api.get(`/sample/${recordId}`)
  return data
}

/**
 * Classify a signal array.
 */
export async function classifySignal(signal, opts = {}) {
  const { data } = await api.post('/classify', {
    signal,
    sampling_rate: opts.samplingRate ?? 360,
    model_type: opts.modelType ?? 'rf',
    include_shap: opts.includeShap ?? true,
    include_llm: opts.includeLlm ?? false,
  })
  return data
}

/**
 * Upload a CSV file for classification.
 */
export async function classifyUpload(file, opts = {}) {
  const form = new FormData()
  form.append('file', file)
  const params = new URLSearchParams({
    sampling_rate: opts.samplingRate ?? 360,
    model_type: opts.modelType ?? 'rf',
    include_shap: opts.includeShap ?? true,
    include_llm: opts.includeLlm ?? false,
  })
  const { data } = await api.post(`/classify/upload?${params}`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

/**
 * Fetch training validation metrics (per-class F1, confusion matrix, patient split).
 */
export async function fetchValidation() {
  const { data } = await api.get('/validation')
  return data
}
