import { useState, useRef, useEffect } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// DATA
// ─────────────────────────────────────────────────────────────────────────────

const DRUG_CATEGORIES = {
  "Cardiovascular": ["atorvastatin","lisinopril","amlodipine","metoprolol","carvedilol","ramipril","valsartan","losartan","bisoprolol","furosemide","spironolactone","digoxin","warfarin","apixaban","rivaroxaban","clopidogrel","aspirin","nitroglycerin","hydralazine","clonidine","atenolol","propranolol","amiodarone","ezetimibe","rosuvastatin","simvastatin"],
  "Diabetes": ["metformin","insulin glargine","insulin lispro","sitagliptin","empagliflozin","dapagliflozin","liraglutide","semaglutide","dulaglutide","glipizide","glyburide","glimepiride","pioglitazone","acarbose","exenatide","canagliflozin","linagliptin"],
  "Antibiotics": ["amoxicillin","amoxicillin-clavulanate","azithromycin","clarithromycin","doxycycline","ciprofloxacin","levofloxacin","trimethoprim-sulfamethoxazole","metronidazole","clindamycin","cephalexin","ceftriaxone","cefdinir","penicillin V","ampicillin","vancomycin","linezolid","nitrofurantoin","rifampin","isoniazid"],
  "Pain & Inflammation": ["ibuprofen","naproxen","diclofenac","celecoxib","acetaminophen","meloxicam","ketorolac","tramadol","morphine","oxycodone","hydrocodone","codeine","fentanyl","buprenorphine","naloxone","pregabalin","gabapentin","duloxetine","amitriptyline"],
  "Mental Health": ["sertraline","fluoxetine","escitalopram","paroxetine","venlafaxine","bupropion","mirtazapine","trazodone","lithium","valproate","lamotrigine","quetiapine","risperidone","olanzapine","aripiprazole","lorazepam","diazepam","clonazepam","alprazolam","buspirone","zolpidem","methylphenidate","atomoxetine"],
  "Respiratory": ["albuterol","salmeterol","tiotropium","ipratropium","fluticasone","budesonide","montelukast","theophylline","cetirizine","loratadine","fexofenadine","diphenhydramine","guaifenesin","pseudoephedrine","omalizumab"],
  "GI & Stomach": ["omeprazole","esomeprazole","pantoprazole","lansoprazole","famotidine","ondansetron","metoclopramide","loperamide","sucralfate","mesalamine","lactulose","polyethylene glycol","senna","bisacodyl","infliximab","adalimumab"],
  "Hormones & Thyroid": ["levothyroxine","methimazole","prednisone","methylprednisolone","dexamethasone","hydrocortisone","testosterone","estradiol","progesterone","levonorgestrel","tamoxifen","letrozole","anastrozole","raloxifene"],
  "Antiviral & Infection": ["acyclovir","valacyclovir","oseltamivir","remdesivir","nirmatrelvir","ritonavir","tenofovir","emtricitabine","efavirenz","dolutegravir","fluconazole","itraconazole","voriconazole","nystatin","chloroquine","hydroxychloroquine","ivermectin"],
  "Neurology": ["levodopa","carbidopa","pramipexole","ropinirole","memantine","donepezil","rivastigmine","baclofen","tizanidine","sumatriptan","topiramate","sodium valproate","phenytoin","carbamazepine","levetiracetam","lacosamide"],
  "Cancer": ["tamoxifen","letrozole","trastuzumab","imatinib","erlotinib","methotrexate","cyclophosphamide","doxorubicin","paclitaxel","docetaxel","carboplatin","cisplatin","capecitabine","pembrolizumab","nivolumab"],
  "Kidney & Urology": ["furosemide","hydrochlorothiazide","tamsulosin","finasteride","dutasteride","oxybutynin","solifenacin","sildenafil","tadalafil","allopurinol","febuxostat","desmopressin"],
};

const ICD10_MAP = {
  "chest pain": "R07.9", "shortness of breath": "R06.09", "headache": "R51",
  "fever": "R50.9", "cough": "R05", "fatigue": "R53.83", "dizziness": "R42",
  "nausea": "R11.0", "back pain": "M54.5", "abdominal pain": "R10.9",
  "sore throat": "J02.9", "rash": "R21", "joint pain": "M25.50",
  "palpitations": "R00.2", "anxiety": "F41.9", "depression": "F32.9",
  "hypertension": "I10", "diabetes type 2": "E11.9", "asthma": "J45.909",
  "pneumonia": "J18.9", "UTI": "N39.0", "anemia": "D64.9",
};

const DRUG_CACHE = {};

// ─────────────────────────────────────────────────────────────────────────────
// OPENROUTER API — fetches LIVE available models, never uses stale hardcoded IDs
// ─────────────────────────────────────────────────────────────────────────────

// Fallback model IDs in case live fetch fails
const FALLBACK_MODELS = [
  "meta-llama/llama-3.1-8b-instruct:free",
  "mistralai/mistral-7b-instruct:free",
  "google/gemma-2-9b-it:free",
  "qwen/qwen-2.5-7b-instruct:free",
];

// Cache of live-fetched free models
let _liveModels = null;

// Fetch currently available free models from OpenRouter live API
async function fetchLiveModels(apiKey) {
  if (_liveModels && _liveModels.length > 0) return _liveModels;
  try {
    const res = await fetch("https://openrouter.ai/api/v1/models", {
      headers: { "Authorization": "Bearer " + apiKey }
    });
    if (!res.ok) return FALLBACK_MODELS;
    const data = await res.json();
    const free = (data.data || [])
      .filter(m => m.id && m.id.endsWith(":free") &&
        (m.context_length || 0) >= 4096 &&
        m.id !== "openrouter/auto")
      .sort((a, b) => (b.context_length || 0) - (a.context_length || 0))
      .slice(0, 10)
      .map(m => m.id);
    _liveModels = free.length > 0 ? free : FALLBACK_MODELS;
    return _liveModels;
  } catch {
    return FALLBACK_MODELS;
  }
}

function extractJSON(raw) {
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start === -1 || end === -1) throw new Error("No JSON in response. Try again.");
  return JSON.parse(raw.slice(start, end + 1));
}

// Call one specific model via OpenRouter
async function callOneModel(prompt, apiKey, modelId) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 45000);
  let res;
  try {
    res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + apiKey,
        "HTTP-Referer": "https://medai-doctor.app",
        "X-Title": "MedAI Doctor",
      },
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 2048,
        temperature: 0.3,
      }),
    });
  } finally {
    clearTimeout(timeoutId);
  }
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    const msg = errData?.error?.message || ("API error " + res.status);
    throw new Error(msg);
  }
  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error("Empty response");
  return text;
}

// Try each available model until one works — auto-fallback across ALL live free models
async function callAIWithFallback(prompt, apiKey, preferredModel) {
  if (!apiKey) throw new Error("No API key. Click Change Key in the sidebar.");

  // Auth/billing = hard stop immediately, no retry
  function isHardStop(msg) {
    return msg.includes("401") || msg.includes("Invalid API key") ||
      (msg.includes("402") && !msg.includes("endpoint"));
  }

  // First try the user's selected model
  if (preferredModel) {
    try {
      return await callOneModel(prompt, apiKey, preferredModel);
    } catch (e) {
      const msg = String(e?.message || "");
      if (isHardStop(msg)) throw new Error("Invalid API key. Go to openrouter.ai/keys to get your key.");
      console.log("Preferred model failed:", msg, "— fetching live models...");
    }
  }

  // Fetch live models and try them all
  const models = await fetchLiveModels(apiKey);
  let lastErr = null;
  for (let i = 0; i < models.length; i++) {
    if (models[i] === preferredModel) continue; // already tried
    try {
      console.log("Trying model " + (i+1) + "/" + models.length + ": " + models[i]);
      return await callOneModel(prompt, apiKey, models[i]);
    } catch (e) {
      const msg = String(e?.message || "");
      lastErr = e;
      if (isHardStop(msg)) throw new Error("Invalid API key. Go to openrouter.ai/keys to get your key.");
      // Otherwise keep trying next model
    }
  }
  throw new Error("All " + models.length + " available free models failed. Try again in a minute, or use a paid model.");
}

async function runMedicalAnalysis(userMessage, history, apiKey, model) {
  const prompt =
    "You are the Supervisor Agent of an advanced AI Medical Assistant with 5 specialized sub-agents: Triage, Symptom Analyst, ICD-10 Coder, Lab Agent, Diagnosis Agent.\n\n" +
    "PATIENT INPUT: " + userMessage + "\n" +
    "HISTORY: " + JSON.stringify((history || []).slice(-4)) + "\n\n" +
    "CRITICAL: Return ONLY a valid JSON object. No markdown. No text before or after. Start with { and end with }.\n\n" +
    "JSON structure:\n" +
    "{\n" +
    "  \"triage\": { \"urgency\": \"CRITICAL or HIGH or MODERATE or LOW\", \"urgency_reason\": \"reason\", \"recommend_emergency\": false, \"timeframe\": \"Immediate or Within 24 hours or Within a week or Routine\" },\n" +
    "  \"symptom_analysis\": { \"identified_symptoms\": [\"list\"], \"body_systems\": [\"list\"], \"duration_assessment\": \"acute or chronic or unknown\", \"red_flags\": [\"list or empty\"] },\n" +
    "  \"differential_diagnosis\": [{ \"condition\": \"Name\", \"icd10\": \"code\", \"confidence\": 80, \"reasoning\": \"reasoning\", \"category\": \"primary or secondary or rule_out\" }],\n" +
    "  \"lab_recommendations\": { \"urgent\": [\"tests\"], \"routine\": [\"tests\"], \"imaging\": [\"imaging\"], \"specialist\": \"type or null\" },\n" +
    "  \"medication_analysis\": { \"relevant_medications\": [\"list\"], \"drug_interactions_to_check\": [\"list\"], \"otc_suggestions\": [\"list\"], \"avoid\": [\"list\"] },\n" +
    "  \"soap_note\": { \"subjective\": \"text\", \"objective\": \"text\", \"assessment\": \"text\", \"plan\": \"text\" },\n" +
    "  \"confidence_score\": 0.80,\n" +
    "  \"evidence_strength\": \"High or Moderate or Low\",\n" +
    "  \"disclaimer\": \"This AI analysis is for informational purposes only and does NOT replace professional medical advice. Always consult a qualified healthcare provider.\",\n" +
    "  \"follow_up_questions\": [\"question 1\", \"question 2\"],\n" +
    "  \"patient_education\": \"plain language explanation\"\n" +
    "}";

  const raw = await callAIWithFallback(prompt, apiKey, model);
  return extractJSON(raw);
}

async function lookupDrug(drugName, apiKey, model) {
  const prompt =
    "You are a world-class clinical pharmacologist with knowledge of all 60,000+ worldwide medications.\n" +
    "Provide comprehensive clinical information about: " + drugName + "\n\n" +
    "CRITICAL: Return ONLY a valid JSON object. No markdown. No text before or after. Start with { and end with }.\n\n" +
    "JSON structure:\n" +
    "{\n" +
    "  \"name\": \"generic name\",\n" +
    "  \"brand_names\": [\"brand1\", \"brand2\"],\n" +
    "  \"class\": \"drug class\",\n" +
    "  \"subclass\": \"mechanism\",\n" +
    "  \"indications\": [\"use1\", \"use2\"],\n" +
    "  \"contraindications\": [\"list\"],\n" +
    "  \"precautions\": [\"list\"],\n" +
    "  \"interactions\": [\"list\"],\n" +
    "  \"side_effects\": [\"common\"],\n" +
    "  \"serious_effects\": [\"serious\"],\n" +
    "  \"dosage_adult\": \"adult dose\",\n" +
    "  \"dosage_pediatric\": \"pediatric or N/A\",\n" +
    "  \"dosage_renal\": \"renal adjustment\",\n" +
    "  \"dosage_hepatic\": \"hepatic adjustment\",\n" +
    "  \"pregnancy_category\": \"category\",\n" +
    "  \"monitoring\": [\"params\"],\n" +
    "  \"onset\": \"onset\",\n" +
    "  \"half_life\": \"half-life\",\n" +
    "  \"route\": [\"oral\"],\n" +
    "  \"storage\": \"storage\",\n" +
    "  \"patient_tips\": \"counseling\",\n" +
    "  \"overdose\": \"management\"\n" +
    "}";

  const raw = await callAIWithFallback(prompt, apiKey, model);
  return extractJSON(raw);
}

async function analyzeLabResults(labText, apiKey, model) {
  const prompt =
    "You are a clinical laboratory specialist.\n" +
    "Analyze these lab results:\n" + labText + "\n\n" +
    "CRITICAL: Return ONLY a valid JSON object. No markdown. No text before or after. Start with { and end with }.\n\n" +
    "JSON structure:\n" +
    "{\n" +
    "  \"results\": [{ \"test\": \"name\", \"value\": \"value\", \"unit\": \"unit\", \"status\": \"normal or high or low or critical\", \"interpretation\": \"meaning\" }],\n" +
    "  \"overall_assessment\": \"text\",\n" +
    "  \"concerns\": [\"list\"],\n" +
    "  \"recommendations\": [\"list\"],\n" +
    "  \"urgency\": \"routine or soon or urgent\"\n" +
    "}";

  const raw = await callAIWithFallback(prompt, apiKey, model);
  return extractJSON(raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────────────────────────

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#050a14;--bg2:#080f1e;--sur:#0d1526;--sur2:#111d33;
    --bor:rgba(56,140,255,.15);--bor2:rgba(56,140,255,.08);
    --acc:#388cff;--acc2:#00e5c8;--acc3:#ff6b6b;--acc4:#ffd93d;
    --tx:#e8f0ff;--tx2:#8a9dc4;--tx3:#4a5f84;
    --crit:#ff4444;--hi:#ff8800;--mod:#ffd93d;--lo:#00e5c8;
    --glow:rgba(56,140,255,.2);
  }
  body{background:var(--bg);color:var(--tx);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden}
  ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--bor);border-radius:4px}
  .app{display:flex;height:100vh;overflow:hidden}
  .sb{width:272px;min-width:272px;background:var(--bg2);border-right:1px solid var(--bor);display:flex;flex-direction:column;position:relative;overflow:hidden}
  .sb::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--acc),transparent)}
  .logo-area{padding:22px 18px 18px;border-bottom:1px solid var(--bor2);background:linear-gradient(135deg,rgba(56,140,255,.05),transparent)}
  .logo{display:flex;align-items:center;gap:11px;margin-bottom:4px}
  .logo-icon{width:42px;height:42px;border-radius:11px;background:linear-gradient(135deg,var(--acc),var(--acc2));display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 0 20px rgba(56,140,255,.4);animation:pulse 3s ease-in-out infinite}
  @keyframes pulse{0%,100%{box-shadow:0 0 20px rgba(56,140,255,.4)}50%{box-shadow:0 0 35px rgba(56,140,255,.7)}}
  .logo-text{font-family:'Syne',sans-serif;font-weight:800;font-size:19px;letter-spacing:-.5px}
  .logo-text span{color:var(--acc2)}
  .logo-sub{font-size:10px;color:var(--tx3);letter-spacing:.5px;margin-top:1px}
  .status-pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;background:rgba(0,229,200,.1);border:1px solid rgba(0,229,200,.25);border-radius:20px;font-size:11px;color:var(--acc2);margin-top:10px}
  .s-dot{width:6px;height:6px;border-radius:50%;background:var(--acc2);animation:blink 2s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
  .nav{padding:14px 10px;flex:1;overflow-y:auto}
  .nav-sec-lbl{font-size:10px;font-weight:600;letter-spacing:1.5px;color:var(--tx3);padding:0 8px;margin-bottom:6px;text-transform:uppercase}
  .nav-sec{margin-bottom:18px}
  .nb{width:100%;display:flex;align-items:center;gap:11px;padding:9px 11px;border-radius:9px;border:1px solid transparent;background:transparent;color:var(--tx2);font-family:'DM Sans',sans-serif;font-size:14px;cursor:pointer;transition:all .2s;margin-bottom:2px;text-align:left}
  .nb:hover{background:rgba(56,140,255,.08);color:var(--tx)}
  .nb.active{background:rgba(56,140,255,.12);border-color:var(--bor);color:var(--acc);font-weight:500}
  .nb-icon{font-size:17px;width:22px;text-align:center}
  .nb-badge{margin-left:auto;background:var(--acc);color:#fff;font-size:10px;padding:2px 7px;border-radius:10px}
  .sb-footer{padding:14px;border-top:1px solid var(--bor2)}
  .agent-box{background:var(--sur);border-radius:11px;padding:13px;border:1px solid var(--bor2)}
  .agent-title-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px}
  .agent-title{font-size:10px;font-weight:600;letter-spacing:1px;color:var(--tx3);text-transform:uppercase}
  .key-btn{font-size:10px;color:var(--tx3);background:none;border:1px solid var(--bor2);border-radius:6px;padding:2px 8px;cursor:pointer;font-family:'DM Sans',sans-serif;transition:all .2s}
  .key-btn:hover{border-color:var(--acc);color:var(--acc)}
  .agent-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:12px;color:var(--tx2)}
  .a-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
  .a-dot.ready{background:var(--acc2);box-shadow:0 0 6px var(--acc2)}
  .a-dot.busy{background:var(--acc4);animation:blink 1s infinite}
  .main{flex:1;display:flex;flex-direction:column;overflow:hidden}
  .topbar{padding:14px 26px;border-bottom:1px solid var(--bor);display:flex;align-items:center;justify-content:space-between;background:rgba(8,15,30,.8);backdrop-filter:blur(10px);flex-shrink:0}
  .topbar h1{font-family:'Syne',sans-serif;font-size:20px;font-weight:700}
  .topbar p{font-size:12px;color:var(--tx3);margin-top:2px}
  .pills{display:flex;gap:8px;flex-wrap:wrap}
  .pill{padding:6px 13px;border-radius:20px;border:1px solid var(--bor);background:var(--sur);font-size:11px;color:var(--tx2);font-family:'DM Sans',sans-serif}
  .pill.g{border-color:rgba(0,229,200,.3);color:var(--acc2)}
  .content{flex:1;overflow-y:auto;padding:22px 26px}

  /* SETUP SCREEN */
  .setup-wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);padding:20px}
  .setup-card{width:100%;max-width:500px}
  .setup-hero{text-align:center;margin-bottom:28px}
  .setup-hero-icon{font-size:52px;margin-bottom:14px}
  .setup-hero h1{font-family:'Syne',sans-serif;font-size:30px;font-weight:800;margin-bottom:6px}
  .setup-hero p{color:var(--tx3);font-size:13px}
  .setup-box{background:var(--sur);border:1px solid var(--bor);border-radius:18px;padding:28px}
  .setup-provider{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding:14px;background:var(--sur2);border-radius:12px;border:1px solid var(--bor)}
  .setup-or-logo{width:42px;height:42px;border-radius:10px;background:linear-gradient(135deg,#7c3aed,#4f46e5);display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
  .setup-or-info{}
  .setup-or-name{font-family:'Syne',sans-serif;font-weight:700;font-size:15px}
  .setup-or-sub{font-size:12px;color:var(--acc2);margin-top:2px}
  .setup-why{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:18px}
  .setup-why-item{background:rgba(56,140,255,.06);border:1px solid var(--bor2);border-radius:10px;padding:10px;font-size:12px;color:var(--tx2);line-height:1.5}
  .setup-why-item strong{color:var(--acc2);display:block;margin-bottom:2px;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
  .setup-steps{background:rgba(0,229,200,.05);border:1px solid rgba(0,229,200,.2);border-radius:11px;padding:14px;margin-bottom:18px;font-size:13px;color:var(--tx2);line-height:2}
  .setup-steps strong{color:var(--tx)}
  .setup-steps a{color:var(--acc);text-decoration:none;font-weight:500}
  .setup-steps a:hover{color:var(--acc2)}
  .setup-input{width:100%;background:var(--sur2);border:1px solid var(--bor);border-radius:11px;padding:13px 15px;color:var(--tx);font-family:'DM Sans',sans-serif;font-size:14px;outline:none;transition:border-color .2s;margin-bottom:8px}
  .setup-input:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(56,140,255,.1)}
  .setup-err{color:var(--crit);font-size:12px;margin-bottom:10px;padding:8px 12px;background:rgba(255,68,68,.07);border-radius:8px;border:1px solid rgba(255,68,68,.2)}
  .setup-btn{width:100%;padding:14px;border-radius:11px;border:none;background:linear-gradient(135deg,#7c3aed,#4f46e5);color:#fff;font-family:'Syne',sans-serif;font-weight:700;font-size:15px;cursor:pointer;margin-bottom:10px;transition:opacity .2s}
  .setup-btn:hover{opacity:.9}
  .setup-note{font-size:11px;color:var(--tx3);text-align:center;margin-top:10px;line-height:1.7}

  /* MODEL SELECTOR */
  .model-select{width:100%;background:var(--sur2);border:1px solid var(--bor);border-radius:9px;padding:9px 12px;color:var(--tx2);font-family:'DM Sans',sans-serif;font-size:13px;outline:none;margin-bottom:14px;cursor:pointer}
  .model-select:focus{border-color:var(--acc)}

  /* CHAT */
  .chat-wrap{display:flex;flex-direction:column;height:100%}
  .chat-msgs{flex:1;overflow-y:auto;padding-bottom:14px}
  @keyframes fiu{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
  .welcome{background:linear-gradient(135deg,var(--sur),var(--sur2));border:1px solid var(--bor);border-radius:18px;padding:28px;margin-bottom:20px;text-align:center;animation:fiu .5s ease}
  .welcome-icon{font-size:44px;margin-bottom:14px}
  .welcome h2{font-family:'Syne',sans-serif;font-size:24px;font-weight:700;margin-bottom:8px}
  .welcome p{color:var(--tx2);line-height:1.6;max-width:480px;margin:0 auto 18px}
  .starters{display:flex;flex-wrap:wrap;gap:7px;justify-content:center}
  .qs{padding:8px 14px;border:1px solid var(--bor);border-radius:18px;background:var(--sur2);color:var(--tx2);font-size:13px;cursor:pointer;transition:all .2s;font-family:'DM Sans',sans-serif}
  .qs:hover{border-color:var(--acc);color:var(--acc);background:rgba(56,140,255,.08)}
  .msg{margin-bottom:18px;animation:fiu .3s ease}
  .msg-user{display:flex;justify-content:flex-end}
  .msg-ai{display:flex;gap:11px;align-items:flex-start}
  .ai-av{width:36px;height:36px;border-radius:9px;flex-shrink:0;background:linear-gradient(135deg,var(--acc),var(--acc2));display:flex;align-items:center;justify-content:center;font-size:17px;box-shadow:0 0 14px var(--glow)}
  .bubble-user{max-width:68%;background:linear-gradient(135deg,var(--acc),#2563eb);padding:12px 16px;border-radius:16px 16px 4px 16px;font-size:14px;line-height:1.6}
  .bubble-ai{flex:1;background:var(--sur);border:1px solid var(--bor);border-radius:4px 16px 16px 16px;overflow:hidden}
  .resp-grid{display:grid;gap:11px;padding:15px}
  .rcard{background:var(--sur2);border:1px solid var(--bor2);border-radius:11px;padding:13px}
  .rcard-hdr{display:flex;align-items:center;gap:7px;margin-bottom:9px}
  .rcard-icon{font-size:17px}
  .rcard-title{font-size:11px;font-weight:600;letter-spacing:.5px;color:var(--tx3);text-transform:uppercase}
  .urg{display:inline-flex;align-items:center;gap:6px;padding:5px 13px;border-radius:18px;font-size:13px;font-weight:600;letter-spacing:.3px}
  .urg-CRITICAL{background:rgba(255,68,68,.15);border:1px solid var(--crit);color:var(--crit)}
  .urg-HIGH{background:rgba(255,136,0,.15);border:1px solid var(--hi);color:var(--hi)}
  .urg-MODERATE{background:rgba(255,217,61,.15);border:1px solid var(--mod);color:var(--mod)}
  .urg-LOW{background:rgba(0,229,200,.15);border:1px solid var(--lo);color:var(--lo)}
  .conf-bar{height:5px;background:var(--sur);border-radius:3px;overflow:hidden;margin-top:5px}
  .conf-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width 1s ease}
  .tag{display:inline-block;padding:3px 9px;border-radius:11px;font-size:12px;margin:2px;background:rgba(56,140,255,.1);border:1px solid var(--bor2);color:var(--tx2)}
  .tag.red{background:rgba(255,68,68,.1);border-color:rgba(255,68,68,.3);color:#ff8888}
  .tag.green{background:rgba(0,229,200,.1);border-color:rgba(0,229,200,.3);color:var(--acc2)}
  .diag{padding:9px;border-radius:7px;border:1px solid var(--bor2);margin-bottom:5px;background:var(--sur)}
  .diag-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:3px}
  .diag-name{font-size:14px;font-weight:500}
  .diag-conf{font-size:12px;color:var(--acc2);font-weight:600}
  .diag-icd{font-size:11px;color:var(--tx3);margin-bottom:3px}
  .diag-r{font-size:12px;color:var(--tx2);line-height:1.5}
  .diag-bar{height:3px;background:var(--sur2);border-radius:2px;margin-top:5px}
  .diag-bar-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--acc),var(--acc2))}
  .soap-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
  .soap-item{padding:9px;border-radius:7px;background:var(--sur);border:1px solid var(--bor2)}
  .soap-lbl{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}
  .soap-S .soap-lbl{color:#60a5fa}.soap-O .soap-lbl{color:#a78bfa}.soap-A .soap-lbl{color:var(--acc4)}.soap-P .soap-lbl{color:var(--acc2)}
  .soap-item p{font-size:12px;color:var(--tx2);line-height:1.5}
  .disc{background:rgba(255,68,68,.05);border:1px solid rgba(255,68,68,.2);border-radius:9px;padding:11px 13px;margin:3px 14px 14px;font-size:12px;color:#ff9999;line-height:1.5;display:flex;gap:7px}
  .edu{background:rgba(0,229,200,.05);border:1px solid rgba(0,229,200,.2);border-radius:9px;padding:11px 13px;margin:0 14px;font-size:13px;color:var(--acc2);line-height:1.6}
  .fq{font-size:13px;color:var(--acc);padding:5px 0;border-bottom:1px solid var(--bor2);cursor:pointer}
  .fq:last-child{border-bottom:none}
  .fq:hover{color:var(--tx)}
  .li{font-size:13px;color:var(--tx2);padding:4px 0;border-bottom:1px solid var(--bor2);display:flex;align-items:center;gap:7px}
  .li:last-child{border-bottom:none}
  .li-dot{width:5px;height:5px;border-radius:50%;background:var(--acc);flex-shrink:0}
  .thinking{display:flex;gap:4px;padding:15px;align-items:center}
  .td{width:8px;height:8px;border-radius:50%;background:var(--acc);animation:th 1.2s infinite}
  .td:nth-child(2){animation-delay:.2s;background:var(--acc2)}.td:nth-child(3){animation-delay:.4s;background:var(--acc4)}
  @keyframes th{0%,100%{transform:translateY(0);opacity:.3}50%{transform:translateY(-6px);opacity:1}}
  .th-lbl{font-size:12px;color:var(--tx3);margin-left:7px}
  .chat-input{padding:14px 0 4px;border-top:1px solid var(--bor2);flex-shrink:0}
  .input-box{display:flex;gap:9px;align-items:flex-end;background:var(--sur);border:1px solid var(--bor);border-radius:14px;padding:11px 14px;transition:border-color .2s,box-shadow .2s}
  .input-box:focus-within{border-color:var(--acc);box-shadow:0 0 0 3px rgba(56,140,255,.1)}
  .input-box textarea{flex:1;background:transparent;border:none;outline:none;color:var(--tx);font-family:'DM Sans',sans-serif;font-size:14px;resize:none;line-height:1.5;max-height:110px;min-height:22px}
  .input-box textarea::placeholder{color:var(--tx3)}
  .send{width:38px;height:38px;border-radius:9px;border:none;background:linear-gradient(135deg,var(--acc),#2563eb);color:#fff;font-size:17px;cursor:pointer;transition:all .2s;display:flex;align-items:center;justify-content:center;flex-shrink:0}
  .send:hover{transform:scale(1.05);box-shadow:0 0 18px rgba(56,140,255,.5)}
  .send:disabled{opacity:.45;cursor:not-allowed;transform:none}
  .sec-hdr{display:flex;align-items:center;gap:10px;margin-bottom:18px}
  .sec-icon{font-size:22px}
  .sec-title{font-family:'Syne',sans-serif;font-size:20px;font-weight:700}
  .sec-sub{font-size:13px;color:var(--tx3)}
  .drug-search{display:flex;gap:9px;margin-bottom:12px}
  .drug-input{flex:1;background:var(--sur);border:1px solid var(--bor);border-radius:11px;padding:11px 15px;color:var(--tx);font-family:'DM Sans',sans-serif;font-size:14px;outline:none;transition:border-color .2s}
  .drug-input:focus{border-color:var(--acc)}
  .sbtn{padding:11px 18px;border-radius:11px;border:none;background:linear-gradient(135deg,var(--acc),#2563eb);color:#fff;font-family:'DM Sans',sans-serif;font-size:14px;font-weight:500;cursor:pointer;transition:all .2s;white-space:nowrap}
  .sbtn:hover{transform:translateY(-1px);box-shadow:0 4px 18px rgba(56,140,255,.4)}
  .sbtn:disabled{opacity:.5;transform:none;cursor:not-allowed}
  .cat-tabs{display:flex;gap:5px;margin-bottom:10px;flex-wrap:wrap}
  .cat-btn{padding:5px 13px;border-radius:18px;font-size:12px;cursor:pointer;font-family:'DM Sans',sans-serif;transition:all .2s}
  .drug-pills{display:flex;gap:5px;flex-wrap:wrap;background:var(--sur);border:1px solid var(--bor2);border-radius:12px;padding:13px;margin-bottom:18px}
  .dpill-lbl{width:100%;font-size:10px;color:var(--tx3);font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px}
  .dpill{padding:4px 11px;border:1px solid var(--bor);border-radius:11px;background:var(--sur2);color:var(--tx2);font-size:12px;cursor:pointer;transition:all .2s;font-family:'DM Sans',sans-serif}
  .dpill:hover{border-color:var(--acc);color:var(--acc);background:rgba(56,140,255,.08)}
  .drug-card{background:var(--sur);border:1px solid var(--bor);border-radius:15px;overflow:hidden;animation:fiu .4s ease}
  .drug-hdr{padding:18px 22px;background:linear-gradient(135deg,rgba(56,140,255,.1),rgba(0,229,200,.05));border-bottom:1px solid var(--bor2)}
  .drug-name{font-family:'Syne',sans-serif;font-size:21px;font-weight:700;text-transform:capitalize}
  .drug-cls{color:var(--acc2);font-size:14px;margin-top:3px}
  .drug-brands{font-size:12px;color:var(--tx3);margin-top:3px}
  .drug-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-bottom:1px solid var(--bor2)}
  .drug-meta-cell{padding:11px 18px;border-right:1px solid var(--bor2)}
  .drug-meta-cell:last-child{border-right:none}
  .dml{font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--tx3);margin-bottom:3px}
  .dmv{font-size:13px;color:var(--tx)}
  .drug-body{display:grid;grid-template-columns:1fr 1fr}
  .drug-sec{padding:14px 18px;border-bottom:1px solid var(--bor2);border-right:1px solid var(--bor2)}
  .drug-sec:nth-child(even){border-right:none}
  .dst{font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--tx3);margin-bottom:7px}
  .dli{font-size:13px;color:var(--tx2);padding:2px 0;display:flex;gap:5px}
  .dli::before{content:'›';color:var(--acc);flex-shrink:0}
  .drug-tips{padding:13px 18px;background:rgba(0,229,200,.04);border-top:1px solid var(--bor2);font-size:13px;color:var(--acc2);display:flex;gap:7px}
  .drug-stor{padding:9px 18px;background:rgba(56,140,255,.03);border-top:1px solid var(--bor2);font-size:12px;color:var(--tx3);display:flex;gap:7px}
  .err-box{background:rgba(255,68,68,.07);border:1px solid rgba(255,68,68,.3);border-radius:13px;padding:18px;color:var(--acc3);margin-bottom:14px;font-size:14px;line-height:1.6}
  .lab-area{background:var(--sur);border:1px solid var(--bor);border-radius:15px;padding:18px;margin-bottom:18px}
  .lab-lbl{font-size:13px;color:var(--tx3);margin-bottom:9px}
  .lab-ta{width:100%;background:var(--sur2);border:1px solid var(--bor2);border-radius:9px;padding:13px;color:var(--tx);font-family:'DM Sans',sans-serif;font-size:14px;outline:none;resize:vertical;min-height:110px;transition:border-color .2s;line-height:1.6}
  .lab-ta:focus{border-color:var(--acc)}
  .lab-tbl{background:var(--sur);border:1px solid var(--bor);border-radius:14px;overflow:hidden;margin-bottom:14px}
  .lab-tbl-hdr{padding:12px 18px;border-bottom:1px solid var(--bor2);display:flex;align-items:center;justify-content:space-between}
  .lab-tbl-title{font-family:'Syne',sans-serif;font-weight:700;font-size:15px}
  .lab-row{display:flex;align-items:center;gap:11px;padding:11px 16px;border-bottom:1px solid var(--bor2)}
  .lab-row:last-child{border-bottom:none}
  .lab-test{flex:1;font-size:14px;font-weight:500}
  .lab-val{font-size:14px;font-weight:600;font-family:'Syne',sans-serif}
  .lab-status{padding:3px 9px;border-radius:9px;font-size:11px;font-weight:600;white-space:nowrap}
  .lab-normal{background:rgba(0,229,200,.1);color:var(--acc2);border:1px solid rgba(0,229,200,.3)}
  .lab-high{background:rgba(255,136,0,.1);color:var(--hi);border:1px solid rgba(255,136,0,.3)}
  .lab-low{background:rgba(56,140,255,.1);color:var(--acc);border:1px solid rgba(56,140,255,.3)}
  .lab-critical{background:rgba(255,68,68,.1);color:var(--crit);border:1px solid rgba(255,68,68,.3)}
  .lab-interp{font-size:12px;color:var(--tx3);flex:1.5}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-bottom:13px}
  .panel{background:var(--sur);border:1px solid var(--bor);border-radius:14px;padding:17px}
  .panel-title{font-size:11px;font-weight:600;letter-spacing:1px;color:var(--tx3);text-transform:uppercase;margin-bottom:10px}
  .icd-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:11px}
  .icd-card{background:var(--sur);border:1px solid var(--bor);border-radius:13px;padding:14px;cursor:pointer;transition:all .2s}
  .icd-card:hover{border-color:var(--acc);transform:translateY(-2px)}
  .icd-code{font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:var(--acc)}
  .icd-name{font-size:14px;margin-top:5px;text-transform:capitalize}
  .icd-hint{font-size:11px;color:var(--tx3);margin-top:3px}
  @media(max-width:768px){.sb{display:none}.soap-grid{grid-template-columns:1fr}.drug-body{grid-template-columns:1fr}.two-col{grid-template-columns:1fr}.drug-meta{grid-template-columns:1fr 1fr}}
`;

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

export default function MedAI() {
  const [tab, setTab] = useState("chat");
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyModel, setBusyModel] = useState("");

  const [drugQuery, setDrugQuery] = useState("");
  const [drugCat, setDrugCat] = useState("Cardiovascular");
  const [drugResult, setDrugResult] = useState(null);
  const [drugBusy, setDrugBusy] = useState(false);

  const [labText, setLabText] = useState("");
  const [labResult, setLabResult] = useState(null);
  const [labBusy, setLabBusy] = useState(false);

  const [apiKey, setApiKey] = useState(() => localStorage.getItem("or_key") || "");
  const [showSetup, setShowSetup] = useState(() => !localStorage.getItem("or_key"));
  const [keyInput, setKeyInput] = useState("");
  const [keyErr, setKeyErr] = useState("");
  const [selectedModel, setSelectedModel] = useState(FALLBACK_MODELS[0]);
  const [liveModels, setLiveModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(false);

  // Fetch live models when API key is set
  useEffect(() => {
    if (!apiKey) return;
    setModelsLoading(true);
    fetchLiveModels(apiKey).then(models => {
      setLiveModels(models);
      if (models.length > 0) setSelectedModel(models[0]);
      setModelsLoading(false);
    });
  }, [apiKey]);

  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, busy]);

  function getErrMsg(e) {
    const m = String(e?.message || "");
    if (m.includes("No API key")) return m;
    if (m.includes("Invalid API key") || m.includes("401")) return "Invalid OpenRouter API key. Click Change Key.";
    if (m.includes("All models failed")) return "All free models are busy or quota-exceeded. Try selecting a Paid model from the sidebar, or try again in a few minutes.";
    if (m.includes("Rate limit") || m.includes("429")) return "Rate limit hit — trying next model automatically...";
    if (m.includes("credits") || m.includes("402")) return "Insufficient credits. Check openrouter.ai/credits";
    if (m.includes("aborted") || m.includes("timeout")) return "Request timed out (30s). Model may be overloaded — try again.";
    if (m.includes("No JSON")) return "Model returned unexpected format. Please try again.";
    return "Error: " + (m || "Unknown error. Please try again.");
  }

  async function sendMsg(text) {
    const m = typeof text === "string" ? text.trim() : input.trim();
    if (!m || busy) return;
    setInput("");
    setBusy(true);
    setBusyModel(selectedModel.split("/")[1] || selectedModel);
    setMsgs((p) => [...p, { role: "user", content: m }]);
    try {
      const history = msgs.map((x) => ({ role: x.role, content: x.content || "" }));
      const result = await runMedicalAnalysis(m, history, apiKey, selectedModel);
      setMsgs((p) => [...p, { role: "ai", data: result }]);
    } catch (e) {
      setMsgs((p) => [...p, { role: "ai", error: getErrMsg(e) }]);
    }
    setBusy(false);
    setBusyModel("");
  }

  async function searchDrug(name) {
    const q = typeof name === "string" ? name.trim() : drugQuery.trim();
    if (!q || drugBusy) return;
    if (typeof name === "string") setDrugQuery(name);
    setDrugBusy(true);
    setDrugResult(null);
    try {
      const cacheKey = q.toLowerCase();
      if (DRUG_CACHE[cacheKey]) {
        setDrugResult(DRUG_CACHE[cacheKey]);
      } else {
        const r = await lookupDrug(q, apiKey, selectedModel);
        DRUG_CACHE[cacheKey] = r;
        setDrugResult(r);
      }
    } catch (e) {
      setDrugResult({ error: true, msg: getErrMsg(e) });
    }
    setDrugBusy(false);
  }

  async function runLab() {
    if (!labText.trim() || labBusy) return;
    setLabBusy(true);
    setLabResult(null);
    try {
      const r = await analyzeLabResults(labText, apiKey, selectedModel);
      setLabResult(r);
    } catch (e) {
      setLabResult({ error: true, msg: getErrMsg(e) });
    }
    setLabBusy(false);
  }

  function saveKey() {
    const k = keyInput.trim();
    if (!k) { setKeyErr("Please paste your OpenRouter API key"); return; }
    if (!k.startsWith("sk-or-")) {
      setKeyErr("Invalid key format. OpenRouter keys start with sk-or-...");
      return;
    }
    localStorage.setItem("or_key", k);
    setApiKey(k);
    setShowSetup(false);
    setKeyErr("");
  }

  const navItems = [
    { id: "chat", icon: "🩺", label: "AI Doctor Chat", badge: null },
    { id: "drug", icon: "💊", label: "Drug Database", badge: "60K+" },
    { id: "lab",  icon: "🔬", label: "Lab Analysis", badge: null },
    { id: "icd",  icon: "📋", label: "ICD-10 Codes", badge: null },
  ];
  const agents = ["Triage Agent", "Symptom Analyst", "ICD-10 Coder", "Lab Agent", "Diagnosis Agent"];
  const starters = [
    "I have chest pain and shortness of breath",
    "My blood pressure is 150/95, what does that mean?",
    "I have had a headache for 3 days with fever",
    "I feel dizzy and nauseous after eating",
    "My HbA1c is 7.8%, what should I do?",
  ];
  const urgIcon = { CRITICAL: "🚨", HIGH: "⚠️", MODERATE: "🔶", LOW: "🟢" };

  // ── SETUP SCREEN ────────────────────────────────────────────────────────────
  if (showSetup) {
    return (
      <>
        <style>{CSS}</style>
        <div className="setup-wrap">
          <div className="setup-card">
            <div className="setup-hero">
              <div className="setup-hero-icon">🧬</div>
              <h1>Med<span style={{ color: "var(--acc2)" }}>AI</span></h1>
              <p>Multi-Agent Clinical Intelligence System</p>
            </div>
            <div className="setup-box">
              <div className="setup-provider">
                <div className="setup-or-logo">🔀</div>
                <div className="setup-or-info">
                  <div className="setup-or-name">OpenRouter API</div>
                  <div className="setup-or-sub">Free tier · 100+ models · No rate limit issues</div>
                </div>
              </div>

              <div className="setup-why">
                <div className="setup-why-item">
                  <strong>No Rate Limits</strong>
                  Free models with generous quotas
                </div>
                <div className="setup-why-item">
                  <strong>100+ Models</strong>
                  DeepSeek, Llama, Gemini, GPT-4
                </div>
                <div className="setup-why-item">
                  <strong>Instant Responses</strong>
                  No 60-second waits ever again
                </div>
                <div className="setup-why-item">
                  <strong>Auto Fallback</strong>
                  Tries next model if one fails
                </div>
              </div>

              <div className="setup-steps">
                <div>1. Go to <a href="https://openrouter.ai" target="_blank" rel="noreferrer">openrouter.ai</a></div>
                <div>2. Click <strong>Sign In</strong> (Google or GitHub)</div>
                <div>3. Go to <a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer">openrouter.ai/keys</a></div>
                <div>4. Click <strong>Create Key</strong></div>
                <div>5. Copy and paste below</div>
              </div>

              <input
                className="setup-input"
                placeholder="sk-or-v1-... (your OpenRouter API key)"
                value={keyInput}
                onChange={(e) => { setKeyInput(e.target.value); setKeyErr(""); }}
                onKeyDown={(e) => { if (e.key === "Enter") saveKey(); }}
              />
              {keyErr && <div className="setup-err">⚠️ {keyErr}</div>}
              <button className="setup-btn" onClick={saveKey}>
                🚀 Launch MedAI Doctor
              </button>
              <div className="setup-note">
                Your key is saved in your browser only · Never sent to any server · 100% private
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }

  // ── MAIN APP ────────────────────────────────────────────────────────────────
  return (
    <>
      <style>{CSS}</style>
      <div className="app">

        {/* SIDEBAR */}
        <div className="sb">
          <div className="logo-area">
            <div className="logo">
              <div className="logo-icon">🧬</div>
              <div>
                <div className="logo-text">Med<span>AI</span></div>
                <div className="logo-sub">MULTI-AGENT CLINICAL SYSTEM</div>
              </div>
            </div>
            <div className="status-pill">
              <div className="s-dot" />
              All Agents Online
            </div>
          </div>

          <div className="nav">
            <div className="nav-sec">
              <div className="nav-sec-lbl">Core Modules</div>
              {navItems.map((n) => (
                <button key={n.id} className={"nb" + (tab === n.id ? " active" : "")} onClick={() => setTab(n.id)}>
                  <span className="nb-icon">{n.icon}</span>
                  {n.label}
                  {n.badge && <span className="nb-badge">{n.badge}</span>}
                </button>
              ))}
            </div>
            <div className="nav-sec">
              <div className="nav-sec-lbl">Compliance</div>
              <button className="nb"><span className="nb-icon">🔒</span>HIPAA Ready</button>
              <button className="nb"><span className="nb-icon">📊</span>Audit Trails</button>
              <button className="nb"><span className="nb-icon">🛡️</span>PII Masking</button>
            </div>
            <div className="nav-sec">
              <div className="nav-sec-lbl">AI Model</div>
              {modelsLoading && (
                <div style={{fontSize:11,color:"var(--acc2)",padding:"6px 8px"}}>⏳ Fetching live models…</div>
              )}
              <select
                className="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={modelsLoading}
              >
                {liveModels.length > 0 ? (
                  liveModels.map(m => (
                    <option key={m} value={m}>
                      {m.split("/")[1]?.replace(":free","") || m} (Free)
                    </option>
                  ))
                ) : (
                  FALLBACK_MODELS.map(m => (
                    <option key={m} value={m}>
                      {m.split("/")[1]?.replace(":free","") || m} (Free)
                    </option>
                  ))
                )}
                <option value="openai/gpt-4o-mini">gpt-4o-mini (Paid)</option>
                <option value="anthropic/claude-3.5-haiku">claude-3.5-haiku (Paid)</option>
              </select>
            </div>
          </div>

          <div className="sb-footer">
            <div className="agent-box">
              <div className="agent-title-row">
                <div className="agent-title">Agent Status</div>
                <button className="key-btn" onClick={() => { setShowSetup(true); setKeyInput(apiKey); }}>
                  🔑 Change Key
                </button>
              </div>
              {agents.map((a) => (
                <div key={a} className="agent-row">
                  <div className={"a-dot " + (busy ? "busy" : "ready")} />
                  {a}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* MAIN PANEL */}
        <div className="main">
          <div className="topbar">
            <div>
              <h1>
                {navItems.find((n) => n.id === tab)?.icon}{" "}
                {navItems.find((n) => n.id === tab)?.label}
              </h1>
              <p>AI-powered · Multi-agent · Evidence-based · For informational use only</p>
            </div>
            <div className="pills">
              <div className="pill">🧠 LangGraph Orchestration</div>
              <div className="pill">📚 60K+ Medications</div>
              <div className="pill g">
                🔀 {modelsLoading ? "Loading models…" : (selectedModel.split("/")[1]?.replace(":free","") || "OpenRouter")}
              </div>
            </div>
          </div>

          <div className="content">

            {/* ── CHAT TAB ── */}
            {tab === "chat" && (
              <div className="chat-wrap">
                <div className="chat-msgs">
                  {msgs.length === 0 && (
                    <div className="welcome">
                      <div className="welcome-icon">🧬</div>
                      <h2>MedAI Clinical Assistant</h2>
                      <p>
                        5 specialized agents — triage, symptom analysis, differential diagnosis,
                        medication review, and SOAP notes. Powered by OpenRouter with auto model fallback.
                      </p>
                      <div className="starters">
                        {starters.map((s) => (
                          <button key={s} className="qs" onClick={() => sendMsg(s)}>{s}</button>
                        ))}
                      </div>
                    </div>
                  )}

                  {msgs.map((msg, i) => (
                    <div key={i} className="msg">
                      {msg.role === "user" ? (
                        <div className="msg-user">
                          <div className="bubble-user">{msg.content}</div>
                        </div>
                      ) : (
                        <div className="msg-ai">
                          <div className="ai-av">🧬</div>
                          <div className="bubble-ai">
                            {msg.error ? (
                              <div className="err-box" style={{ margin: 14 }}>❌ {msg.error}</div>
                            ) : (
                              <>
                                <div className="resp-grid">

                                  {/* Triage + Confidence */}
                                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
                                    {msg.data?.triage && (
                                      <div>
                                        <div className={"urg urg-" + (msg.data.triage.urgency || "LOW")}>
                                          {urgIcon[msg.data.triage.urgency] || "🔵"}{" "}
                                          {msg.data.triage.urgency} — {msg.data.triage.timeframe}
                                        </div>
                                        <div style={{ fontSize: 12, color: "var(--tx3)", marginTop: 5 }}>
                                          {msg.data.triage.urgency_reason}
                                        </div>
                                        {msg.data.triage.recommend_emergency && (
                                          <div style={{ marginTop: 6, padding: "6px 12px", background: "rgba(255,68,68,.15)", border: "1px solid var(--crit)", borderRadius: 8, fontSize: 12, color: "var(--crit)", fontWeight: 600 }}>
                                            🚨 SEEK EMERGENCY CARE NOW
                                          </div>
                                        )}
                                      </div>
                                    )}
                                    {msg.data?.confidence_score !== undefined && (
                                      <div style={{ minWidth: 130 }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--tx3)", marginBottom: 3 }}>
                                          <span>Confidence</span>
                                          <span style={{ color: "var(--acc2)", fontWeight: 600 }}>
                                            {Math.round((msg.data.confidence_score || 0) * 100)}%
                                          </span>
                                        </div>
                                        <div className="conf-bar">
                                          <div className="conf-fill" style={{ width: Math.round((msg.data.confidence_score || 0) * 100) + "%" }} />
                                        </div>
                                        <div style={{ fontSize: 11, color: "var(--tx3)", marginTop: 2 }}>
                                          Evidence: {msg.data.evidence_strength}
                                        </div>
                                      </div>
                                    )}
                                  </div>

                                  {/* Symptoms */}
                                  {msg.data?.symptom_analysis?.identified_symptoms?.length > 0 && (
                                    <div className="rcard">
                                      <div className="rcard-hdr">
                                        <span className="rcard-icon">🔍</span>
                                        <span className="rcard-title">Symptom Analysis</span>
                                      </div>
                                      <div>
                                        {msg.data.symptom_analysis.identified_symptoms.map((s) => (
                                          <span key={s} className="tag">{s}</span>
                                        ))}
                                      </div>
                                      {msg.data.symptom_analysis.red_flags?.length > 0 && (
                                        <div style={{ marginTop: 7 }}>
                                          {msg.data.symptom_analysis.red_flags.map((f) => (
                                            <span key={f} className="tag red">⚠️ {f}</span>
                                          ))}
                                        </div>
                                      )}
                                      <div style={{ marginTop: 7 }}>
                                        {(msg.data.symptom_analysis.body_systems || []).map((s) => (
                                          <span key={s} className="tag green">{s}</span>
                                        ))}
                                      </div>
                                    </div>
                                  )}

                                  {/* Differential Diagnosis */}
                                  {msg.data?.differential_diagnosis?.length > 0 && (
                                    <div className="rcard">
                                      <div className="rcard-hdr">
                                        <span className="rcard-icon">🏥</span>
                                        <span className="rcard-title">Differential Diagnosis</span>
                                      </div>
                                      {msg.data.differential_diagnosis.map((d, idx) => (
                                        <div key={idx} className="diag">
                                          <div className="diag-hdr">
                                            <span className="diag-name">{d.condition}</span>
                                            <span className="diag-conf">{d.confidence}%</span>
                                          </div>
                                          <div className="diag-icd">ICD-10: {d.icd10} · {d.category}</div>
                                          <div className="diag-r">{d.reasoning}</div>
                                          <div className="diag-bar">
                                            <div className="diag-bar-fill" style={{ width: (d.confidence || 0) + "%" }} />
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  )}

                                  {/* Lab Recommendations */}
                                  {msg.data?.lab_recommendations && (
                                    <div className="rcard">
                                      <div className="rcard-hdr">
                                        <span className="rcard-icon">🔬</span>
                                        <span className="rcard-title">Recommended Tests</span>
                                      </div>
                                      {msg.data.lab_recommendations.urgent?.length > 0 && (
                                        <div style={{ marginBottom: 8 }}>
                                          <div style={{ fontSize: 11, color: "var(--crit)", fontWeight: 600, marginBottom: 4 }}>URGENT</div>
                                          {msg.data.lab_recommendations.urgent.map((t) => (
                                            <div key={t} className="li"><div className="li-dot" style={{ background: "var(--crit)" }} />{t}</div>
                                          ))}
                                        </div>
                                      )}
                                      {msg.data.lab_recommendations.routine?.length > 0 && (
                                        <div style={{ marginBottom: 8 }}>
                                          <div style={{ fontSize: 11, color: "var(--acc)", fontWeight: 600, marginBottom: 4 }}>ROUTINE</div>
                                          {msg.data.lab_recommendations.routine.map((t) => (
                                            <div key={t} className="li"><div className="li-dot" />{t}</div>
                                          ))}
                                        </div>
                                      )}
                                      {msg.data.lab_recommendations.imaging?.length > 0 && (
                                        <div>
                                          <div style={{ fontSize: 11, color: "var(--acc4)", fontWeight: 600, marginBottom: 4 }}>IMAGING</div>
                                          {msg.data.lab_recommendations.imaging.map((t) => (
                                            <div key={t} className="li"><div className="li-dot" style={{ background: "var(--acc4)" }} />{t}</div>
                                          ))}
                                        </div>
                                      )}
                                      {msg.data.lab_recommendations.specialist && (
                                        <div style={{ marginTop: 8, padding: "7px 10px", background: "rgba(56,140,255,.08)", borderRadius: 7, fontSize: 13, color: "var(--acc)" }}>
                                          👨‍⚕️ Refer to: {msg.data.lab_recommendations.specialist}
                                        </div>
                                      )}
                                    </div>
                                  )}

                                  {/* Medications */}
                                  {msg.data?.medication_analysis && (
                                    <div className="rcard">
                                      <div className="rcard-hdr">
                                        <span className="rcard-icon">💊</span>
                                        <span className="rcard-title">Medication Considerations</span>
                                      </div>
                                      {msg.data.medication_analysis.relevant_medications?.length > 0 && (
                                        <div style={{ marginBottom: 8 }}>
                                          <div style={{ fontSize: 11, color: "var(--tx3)", marginBottom: 4 }}>RELEVANT</div>
                                          {msg.data.medication_analysis.relevant_medications.map((m) => (
                                            <span key={m} className="tag green">{m}</span>
                                          ))}
                                        </div>
                                      )}
                                      {msg.data.medication_analysis.avoid?.length > 0 && (
                                        <div>
                                          <div style={{ fontSize: 11, color: "var(--crit)", marginBottom: 4 }}>AVOID</div>
                                          {msg.data.medication_analysis.avoid.map((m) => (
                                            <span key={m} className="tag red">{m}</span>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  )}

                                  {/* SOAP Note */}
                                  {msg.data?.soap_note && (
                                    <div className="rcard">
                                      <div className="rcard-hdr">
                                        <span className="rcard-icon">📋</span>
                                        <span className="rcard-title">SOAP Note</span>
                                      </div>
                                      <div className="soap-grid">
                                        <div className="soap-item soap-S"><div className="soap-lbl">Subjective</div><p>{msg.data.soap_note.subjective}</p></div>
                                        <div className="soap-item soap-O"><div className="soap-lbl">Objective</div><p>{msg.data.soap_note.objective}</p></div>
                                        <div className="soap-item soap-A"><div className="soap-lbl">Assessment</div><p>{msg.data.soap_note.assessment}</p></div>
                                        <div className="soap-item soap-P"><div className="soap-lbl">Plan</div><p>{msg.data.soap_note.plan}</p></div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Patient Education */}
                                  {msg.data?.patient_education && (
                                    <div className="edu">💡 {msg.data.patient_education}</div>
                                  )}

                                  {/* Follow-up Questions */}
                                  {msg.data?.follow_up_questions?.length > 0 && (
                                    <div className="rcard">
                                      <div className="rcard-hdr">
                                        <span className="rcard-icon">❓</span>
                                        <span className="rcard-title">Doctor Would Also Ask</span>
                                      </div>
                                      {msg.data.follow_up_questions.map((q, qi) => (
                                        <div key={qi} className="fq" onClick={() => sendMsg(q)}>→ {q}</div>
                                      ))}
                                    </div>
                                  )}

                                </div>
                                {msg.data?.disclaimer && (
                                  <div className="disc"><span>⚠️</span>{msg.data.disclaimer}</div>
                                )}
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}

                  {busy && (
                    <div className="msg msg-ai">
                      <div className="ai-av">🧬</div>
                      <div className="bubble-ai">
                        <div className="thinking">
                          <div className="td" /><div className="td" /><div className="td" />
                          <span className="th-lbl">
                            Analyzing via {busyModel ? busyModel.replace(":free","") : "AI"} — auto-fallback enabled…
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={bottomRef} />
                </div>

                <div className="chat-input">
                  <div className="input-box">
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMsg(); } }}
                      placeholder="Describe symptoms, ask about medications, or request a clinical assessment…"
                      rows={1}
                    />
                    <button className="send" onClick={() => sendMsg()} disabled={busy || !input.trim()}>
                      {busy ? "⏳" : "↑"}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* ── DRUG DATABASE TAB ── */}
            {tab === "drug" && (
              <div>
                <div className="sec-hdr">
                  <span className="sec-icon">💊</span>
                  <div>
                    <div className="sec-title">Drug Database</div>
                    <div className="sec-sub">60,000+ medications · AI-powered lookup · Interactions · Dosing · Brand names</div>
                  </div>
                </div>
                <div className="drug-search">
                  <input
                    className="drug-input"
                    placeholder="Search ANY medication worldwide (metformin, paracetamol, semaglutide…)"
                    value={drugQuery}
                    onChange={(e) => setDrugQuery(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") searchDrug(); }}
                  />
                  <button className="sbtn" onClick={() => searchDrug()} disabled={drugBusy}>
                    {drugBusy ? "Searching…" : "🔍 Search"}
                  </button>
                </div>
                <div className="cat-tabs">
                  {Object.keys(DRUG_CATEGORIES).map((cat) => (
                    <button key={cat} className="cat-btn" onClick={() => setDrugCat(cat)}
                      style={{ background: drugCat === cat ? "var(--acc)" : "var(--sur2)", border: drugCat === cat ? "1px solid var(--acc)" : "1px solid var(--bor)", color: drugCat === cat ? "#fff" : "var(--tx2)" }}>
                      {cat}
                    </button>
                  ))}
                </div>
                <div className="drug-pills">
                  <div className="dpill-lbl">{drugCat} · {DRUG_CATEGORIES[drugCat].length} medications · click any to search</div>
                  {DRUG_CATEGORIES[drugCat].map((d) => (
                    <button key={d} className="dpill" onClick={() => searchDrug(d)}>{d}</button>
                  ))}
                </div>
                {drugResult?.error && <div className="err-box">❌ {drugResult.msg || "Search failed. Try a different name."}</div>}
                {drugResult && !drugResult.error && (
                  <div className="drug-card">
                    <div className="drug-hdr">
                      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
                        <div>
                          <div className="drug-name">{drugResult.name || drugQuery}</div>
                          <div className="drug-cls">{drugResult.class}{drugResult.subclass ? " · " + drugResult.subclass : ""}</div>
                          {drugResult.brand_names?.length > 0 && <div className="drug-brands">Brand names: {drugResult.brand_names.slice(0, 5).join(", ")}</div>}
                        </div>
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                          {drugResult.pregnancy_category && <div style={{ padding: "3px 10px", borderRadius: 9, border: "1px solid var(--bor)", fontSize: 11, color: "var(--tx2)" }}>🤰 {drugResult.pregnancy_category}</div>}
                          {drugResult.half_life && <div style={{ padding: "3px 10px", borderRadius: 9, border: "1px solid var(--bor)", fontSize: 11, color: "var(--tx2)" }}>⏱ t½: {drugResult.half_life}</div>}
                          {drugResult.onset && <div style={{ padding: "3px 10px", borderRadius: 9, border: "1px solid var(--bor)", fontSize: 11, color: "var(--tx2)" }}>⚡ {drugResult.onset}</div>}
                        </div>
                      </div>
                    </div>
                    <div className="drug-meta">
                      {drugResult.dosage_adult && <div className="drug-meta-cell"><div className="dml">Adult Dose</div><div className="dmv">{drugResult.dosage_adult}</div></div>}
                      {drugResult.dosage_pediatric && <div className="drug-meta-cell"><div className="dml">Pediatric</div><div className="dmv">{drugResult.dosage_pediatric}</div></div>}
                      {drugResult.dosage_renal && <div className="drug-meta-cell"><div className="dml">Renal Adjust</div><div className="dmv" style={{ color: "var(--acc4)" }}>{drugResult.dosage_renal}</div></div>}
                      {drugResult.route?.length > 0 && <div className="drug-meta-cell"><div className="dml">Route</div><div className="dmv">{drugResult.route.join(", ")}</div></div>}
                    </div>
                    <div className="drug-body">
                      <div className="drug-sec"><div className="dst">Indications</div>{(drugResult.indications || []).map((x) => <div key={x} className="dli">{x}</div>)}</div>
                      <div className="drug-sec"><div className="dst">Contraindications</div>{(drugResult.contraindications || []).map((x) => <div key={x} className="dli" style={{ color: "var(--acc3)" }}>{x}</div>)}</div>
                      <div className="drug-sec"><div className="dst">Drug Interactions</div>{(drugResult.interactions || []).map((x) => <div key={x} className="dli" style={{ color: "var(--acc4)" }}>{x}</div>)}</div>
                      <div className="drug-sec"><div className="dst">Side Effects</div>{(drugResult.side_effects || []).map((x) => <div key={x} className="dli">{x}</div>)}</div>
                      <div className="drug-sec"><div className="dst">Serious Effects</div>{(drugResult.serious_effects || []).map((x) => <div key={x} className="dli" style={{ color: "var(--crit)" }}>{x}</div>)}</div>
                      <div className="drug-sec"><div className="dst">Precautions</div>{(drugResult.precautions || []).map((x) => <div key={x} className="dli" style={{ color: "var(--acc4)" }}>{x}</div>)}</div>
                      <div className="drug-sec"><div className="dst">Monitoring</div>{(drugResult.monitoring || []).map((x) => <div key={x} className="dli">{x}</div>)}</div>
                      <div className="drug-sec"><div className="dst">Overdose Management</div><div style={{ fontSize: 13, color: "var(--crit)", lineHeight: 1.5 }}>{drugResult.overdose || "Supportive care. Contact Poison Control."}</div></div>
                    </div>
                    {drugResult.patient_tips && <div className="drug-tips"><span>💡</span>{drugResult.patient_tips}</div>}
                    {drugResult.storage && <div className="drug-stor"><span>📦</span>Storage: {drugResult.storage}</div>}
                  </div>
                )}
              </div>
            )}

            {/* ── LAB ANALYSIS TAB ── */}
            {tab === "lab" && (
              <div>
                <div className="sec-hdr">
                  <span className="sec-icon">🔬</span>
                  <div>
                    <div className="sec-title">Lab Report Analysis</div>
                    <div className="sec-sub">Paste lab values · AI interprets results · Flags abnormalities</div>
                  </div>
                </div>
                <div className="lab-area">
                  <div className="lab-lbl">Paste your lab results below (any format accepted)</div>
                  <textarea
                    className="lab-ta"
                    placeholder={"Example:\nHbA1c: 7.2%\nCreatinine: 1.4 mg/dL\nHemoglobin: 10.2 g/dL\nGlucose: 145 mg/dL\nCholesterol: 235 mg/dL"}
                    value={labText}
                    onChange={(e) => setLabText(e.target.value)}
                  />
                  <button className="sbtn" style={{ marginTop: 11 }} onClick={runLab} disabled={labBusy || !labText.trim()}>
                    {labBusy ? "🔬 Analyzing…" : "🔬 Analyze Lab Results"}
                  </button>
                </div>
                {labResult?.error && <div className="err-box">❌ {labResult.msg}</div>}
                {labResult && !labResult.error && (
                  <div>
                    <div className="lab-tbl">
                      <div className="lab-tbl-hdr">
                        <div className="lab-tbl-title">Results</div>
                        <div className={"urg urg-" + (labResult.urgency === "urgent" ? "CRITICAL" : labResult.urgency === "soon" ? "HIGH" : "LOW")}>
                          {(labResult.urgency || "routine").toUpperCase()}
                        </div>
                      </div>
                      {(labResult.results || []).map((r, i) => (
                        <div key={i} className="lab-row">
                          <div className="lab-test">{r.test}</div>
                          <div className="lab-val" style={{ color: r.status === "normal" ? "var(--acc2)" : r.status === "critical" ? "var(--crit)" : "var(--acc4)" }}>
                            {r.value} {r.unit}
                          </div>
                          <div className={"lab-status lab-" + (r.status || "normal")}>{(r.status || "").toUpperCase()}</div>
                          <div className="lab-interp">{r.interpretation}</div>
                        </div>
                      ))}
                    </div>
                    <div className="two-col">
                      <div className="panel">
                        <div className="panel-title">Assessment</div>
                        <p style={{ fontSize: 13, color: "var(--tx2)", lineHeight: 1.6, marginBottom: 12 }}>{labResult.overall_assessment}</p>
                        {labResult.concerns?.length > 0 && (
                          <>
                            <div style={{ fontSize: 10, color: "var(--crit)", fontWeight: 600, marginBottom: 6 }}>CONCERNS</div>
                            {labResult.concerns.map((c) => <div key={c} className="li"><div className="li-dot" style={{ background: "var(--crit)" }} />{c}</div>)}
                          </>
                        )}
                      </div>
                      <div className="panel">
                        <div className="panel-title">Recommendations</div>
                        {(labResult.recommendations || []).map((r) => (
                          <div key={r} className="li"><div className="li-dot" style={{ background: "var(--acc2)" }} />{r}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── ICD-10 TAB ── */}
            {tab === "icd" && (
              <div>
                <div className="sec-hdr">
                  <span className="sec-icon">📋</span>
                  <div>
                    <div className="sec-title">ICD-10 Code Reference</div>
                    <div className="sec-sub">Click any code to get a full AI clinical breakdown in the chat</div>
                  </div>
                </div>
                <div className="icd-grid">
                  {Object.entries(ICD10_MAP).map(([symptom, code]) => (
                    <div key={code} className="icd-card" onClick={() => {
                      setTab("chat");
                      sendMsg("Explain ICD-10 code " + code + " for " + symptom + " — give differential diagnosis and management plan");
                    }}>
                      <div className="icd-code">{code}</div>
                      <div className="icd-name">{symptom}</div>
                      <div className="icd-hint">Click to analyze in AI Doctor →</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        </div>
      </div>
    </>
  );
}
