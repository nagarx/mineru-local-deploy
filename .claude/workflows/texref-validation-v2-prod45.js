export const meta = {
  name: 'texref-validation-v2-prod45',
  description: 'Validation v2 over the FULL 45-paper arXiv corpus: three-pass check (display equations, notation collisions, inline maths) against the author-source reference, with mandatory adversarial verification of every finding against TeX and the source PDF.',
  phases: [
    { title: 'Validate', detail: 'one agent per paper — VALIDATION.md v2, three passes' },
    { title: 'Verify', detail: 'adversarial re-check of each finding against TeX + PDF' },
  ],
}

const LIB = '/Users/knight/code_local/to_markdown_experiment/MinerU/local_deploy/library/research_papers'

const SEVERITIES = ['CRITICAL', 'MAJOR', 'NOTE']

const VALIDATOR_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['equations_checked', 'inline_spans_checked', 'unlocatable_count', 'findings', 'notes', 'verdict'],
  properties: {
    equations_checked: { type: 'integer' },
    inline_spans_checked: { type: 'integer' },
    unlocatable_count: { type: 'integer', description: 'reference items you could NOT confidently locate' },
    findings: {
      type: 'array',
      description: 'CRITICAL and MAJOR only. NOTE-level observations go in `notes`.',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['ref_id', 'severity', 'reference_latex', 'markdown_latex', 'defect'],
        properties: {
          ref_id: { type: 'string', description: 'eq-NNNN, in-NNNN, a notation symbol, or ABSENT-FROM-REFERENCE' },
          pass: { type: 'string', enum: ['display', 'notation', 'inline'] },
          md_tag: { type: 'string' },
          section: { type: 'string' },
          severity: { type: 'string', enum: ['CRITICAL', 'MAJOR'] },
          reference_latex: { type: 'string' },
          markdown_latex: { type: 'string' },
          defect: { type: 'string' },
          source: { type: 'string' },
        },
      },
    },
    notes: {
      type: 'array',
      description: 'presentation-only observations; excluded from the verdict',
      items: { type: 'string' },
    },
    verdict: { type: 'string' },
  },
}

const VERIFIER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['finding_index', 'ruling', 'reason'],
        properties: {
          finding_index: { type: 'integer' },
          ruling: { type: 'string', enum: ['REAL_DEFECT', 'NOT_A_DEFECT', 'UNCERTAIN'] },
          agreed_severity: { type: 'string', enum: [...SEVERITIES, 'NONE'] },
          reason: { type: 'string' },
          pdf_evidence: { type: 'string' },
        },
      },
    },
  },
}

function validatorPrompt(p, mdRoot) {
  return `You are verifying that a MinerU-extracted markdown file faithfully represents a research
paper's mathematics. Wrong mathematics here becomes wrong trading code, so precision
matters more than throughput.

**Reference — display equations (authoritative):** ${LIB}/texref/${p.slug}/equations.md
**Reference — inline maths and notation (authoritative):** ${LIB}/texref/${p.slug}/inline.md
**Under test:** ${mdRoot}/${p.file}
**Source of last resort:** ${LIB}/texref/${p.slug}/paper.flat.tex — the author's own LaTeX,
comment-stripped, with the line numbers both references cite.

Use exactly these four files. Do NOT open the PDF — this check is markdown-versus-source by
design, and the PDF is the verifier's evidence, not yours. Do NOT search the filesystem for
other copies of this paper.

The references are derived from the author's LaTeX source, not from the rendered page, so
they carry no OCR or layout-inference risk. Where markdown and reference disagree about
mathematics, the reference is right unless you can demonstrate otherwise from the source.

Note: the reference quotes the author's own macros (\`\\R\`, \`\\bm\`, \`\\mathbbm\`, \`\\dt\`) while
the markdown carries expanded standard LaTeX (\`\\mathbb{R}\`, \`\\pmb\`, \`\\mathbb\`, \`\\Delta\`).
Each entry's \`expanded\` field bridges the two. A difference in macro spelling alone is never
a defect.

### Pass 1 — display equations

Work through \`equations.md\` in order. For each equation:

1. **Locate** the counterpart in the markdown. Use, in this order of reliability: the
   equation **number**, the **section path**, then the **context before/after** prose. Do
   NOT match on formula similarity alone — this corpus contains equations differing by one
   subscript, and 11.6% have a near-identical neighbour. A gap in the markdown's own \`\\tag\`
   sequence is strong evidence an equation was dropped.
2. **Compare** against the \`raw\` block; \`expanded\` is a reading aid.
3. If the equation is **absent** from the markdown, that is a finding.

Then scan the markdown for display equations with **no counterpart** in the reference —
fabricated, promoted from inline, or a table row misread as an equation. Check each against
\`paper.flat.tex\` first. Remember a markdown display block may legitimately correspond to an
inline span in the source, and vice versa; that relocation is not itself a defect.

### Pass 2 — notation

Read the **notation inventory** at the top of \`inline.md\` end to end. It is short, and
ordered so the paper's own notation comes first and standard LaTeX last. Then check the
markdown for **symbol collisions**: a symbol rendered as a *different symbol that also
appears in this paper's inventory with another meaning*. Measured real cases: \`\\mathcal{T}\`
(a tridiagonal matrix) rendered as \`\\tau\`, which the same paper already uses for a time
index; \`\\mathcal{Y}\` (the forecast target) rendered as \`\\mathcal{V}\`; \`\\shuffle\` ⧢ rendered
as \`\\sqcup\` ⊔ — a different operator — eleven times.

**A collision is a defect. A typeface change with no collision is not.**

### Pass 3 — inline maths

\`inline.md\` lists every distinct inline span with an occurrence count. Work through
**"Inline maths with structure"**; each entry carries a subscript, accent, relation or
operator that MinerU can corrupt. Spans under **"Bare symbols"** are covered by Pass 2 — do
not check them one by one.

Spans marked **[table cell]** deserve particular attention: table formulas are routinely
read as plain prose, losing \`\\frac\` bars and altering constants.

Trial 1 found 68 of 118 verified defects in inline maths, prose and table cells, including
5 of 8 criticals. **This pass is where the defects are. Do not rush it.**

### What IS a defect

| severity | test | examples |
|---|---|---|
| **CRITICAL** | the mathematics is **different** | sign changed or inserted · numeric constant altered · term or factor added or dropped · relation changed (\`=\`→\`≤\`, \`<\`→\`≤\`) · operator changed (\`∑\`→\`∏\`, \`·\`→\`+\`, or a different named operator) · conditioning/norm bar lost · integration limit or index origin altered · two equations merged · an equation or row absent · content not in the source at all · a row duplicated |
| **MAJOR** | the mathematics is recoverable but **an implementer could get it wrong** | sub/superscript lost or rescoped · accent invented or dropped where it carries meaning · identifier shattered (\`Logits\`→prose "Logit" + \`s_{k,h}\`) · operator or relation evicted from the maths span (\`\\exp\`→prose \`6\`+\`\\mathrm{xp}\`; \`=\`→U+4E00; \`\\to\` deleted) · a **symbol collision** per Pass 2 · an interval opened or closed (\`(a,b)\`→\`[a,b)\`) · a tensor dimension dropped · prose or a caption spliced into an equation |

### What is NOT a defect — put these in \`notes\`, never in \`findings\`

Every item here was a *refuted* finding in trial 1. Reporting them as defects is the failure
mode that gets the whole check ignored.

- \`\\dfrac\` vs \`\\frac\`, \`\\left(\` vs \`(\`, \`x_{i}\` vs \`x_i\`, \`\\to\` vs \`\\rightarrow\`
- spacing macros (\`\\,\` \`\\;\` \`\\quad\`), row breaks, trailing punctuation
- \`\\begin{cases}\` vs an equivalent \`\\left\\{\\begin{array}\`
- **accent scope**: \`\\overline{AB}\` vs \`\\overline{A}\\,\\overline{B}\` — identical on the page
- **typeface family without collision**: \`\\mathscr{D}\` vs \`\\mathcal{D}\`, \`\\mathtt{T}\` vs
  \`\\mathsf{T}\`, \`\\bm\` vs \`\\pmb\`, \`\\mathbbm\` vs \`\\mathbb\`, boldface lost on a symbol with no
  non-bold counterpart in the inventory
- \`\\_\` in an identifier rendered as \`_\`; HTML \`<sup>\`/\`<sub>\` instead of \`^\`/\`_\`
- en-dash vs minus sign in **prose** (not in maths)
- an equation number carried as \`(8)\` inside the block instead of \`\\tag{8}\`; a missing
  \`\\label\`/\`\\tag\`
- one source \`align\` split across several \`$$\` blocks, or several merged — **unless** the
  split changes which terms belong to which equation

### Unresolved macros

If an entry is marked **⚠ unresolved macros**, that symbol comes from a LaTeX package and its
meaning is not in the source. State that plainly; do not guess what MinerU should have
rendered. But DO report it if MinerU substituted a *different, known* operator.

### Output

\`findings\` holds CRITICAL and MAJOR only. \`notes\` holds presentation observations and does
not count toward the verdict.

Set \`verdict\` to exactly one of:
- \`FAITHFUL — N equations + M inline spans checked, no defects\`
- \`MINOR ISSUES — N checked, X major, no critical\`
- \`UNRELIABLE — N checked, X critical, Y major\`

Three rules that override everything above:

- **Never repair the markdown.** You are measuring, not fixing.
- **If you cannot locate something confidently, say so** (count it in \`unlocatable_count\`)
  rather than guessing at a match.
- **Report what you actually checked.** If you did not get through every entry, say so
  rather than inflating \`equations_checked\` / \`inline_spans_checked\`.

This paper has ${p.ref_equations} display equations and ${p.inline_structural} inline spans
with structure (${p.notation_symbols} notation symbols).`
}

function verifierPrompt(p, findings, mdRoot) {
  return `You are adversarially re-checking claimed defects in a MinerU-extracted markdown file.
Your job is to KILL findings that do not hold up. A false alarm that survives is worse than
a missed defect, because it trains its readers to ignore the check.

Paper: ${p.slug}

Evidence:
- **Author's LaTeX (authoritative for mathematics):** ${LIB}/texref/${p.slug}/equations.md,
  ${LIB}/texref/${p.slug}/inline.md, ${LIB}/texref/${p.slug}/paper.flat.tex
- **The markdown under test:** ${mdRoot}/${p.file}
- **The published PDF (visual arbiter — read the relevant pages):** ${LIB}/inbox/${p.pdf}

For each claimed finding, independently re-derive the answer. **Do not trust the claim's
quoted LaTeX** — re-read both files yourself. Misquotation is a common failure mode; in
trial 1 one finding asserted the exact opposite of what the source said.

- \`NOT_A_DEFECT\` — markdown and source are mathematically equivalent; or the claim misquoted
  a file; or the difference is on the not-a-defect list (\\dfrac vs \\frac, \\left( vs (,
  x_{i} vs x_i, spacing macros, row breaks, trailing punctuation, cases vs an equivalent
  array, accent SCOPE such as \\overline{AB} vs \\overline{A}\\overline{B}, typeface family
  with no collision — \\mathtt vs \\mathsf, \\bm vs \\pmb, \\mathbbm vs \\mathbb, \\mathscr vs
  \\mathcal — \\_ spelling, HTML sup/sub markup, a tag carried as (8) inside the block, one
  align split across several \$\$ blocks); or the reference is wrong and the markdown matches
  the printed PDF.
- \`REAL_DEFECT\` — the markdown states different mathematics, or states it in a way an
  implementer would get wrong. Confirm against the PDF where the PDF can settle it. A
  **symbol collision** — a symbol rendered as a different symbol that also carries meaning in
  this paper — IS a real defect even though a plain typeface change is not.
- \`UNCERTAIN\` — genuinely unsettleable from these sources. Use sparingly; say what is missing.

Set \`agreed_severity\` to the severity YOU judge correct (CRITICAL / MAJOR / NOTE), or NONE
for a refuted finding — the validator's severity is a claim, not an input. Record in
\`pdf_evidence\` what the printed equation actually shows, when you looked.

Return exactly one verdict per finding, keyed by \`finding_index\`.

CLAIMED FINDINGS (index: severity — pass — ref_id — defect):
${findings.map((f, i) => `[${i}] ${f.severity} — ${f.pass || '?'} — ${f.ref_id}${f.md_tag ? ` (md \\tag{${f.md_tag}})` : ''} — ${f.defect}
    claimed reference: ${String(f.reference_latex).slice(0, 400)}
    claimed markdown : ${String(f.markdown_latex).slice(0, 400)}`).join('\n')}`
}

const input = {"md_root":"/private/tmp/claude-501/-Users-knight-code-local-to-markdown-experiment-MinerU/b9134980-fda9-4975-a539-0ce9ace8da02/scratchpad/prod45/md","papers":[{"file":"A_Primer_on_the_Signature_Method_in_Machine_Learning.md","slug":"A Primer on the Signature Method in Machine Learning","pdf":"A Primer on the Signature Method in Machine Learning.pdf","ref_equations":134,"inline_structural":496,"notation_symbols":83},{"file":"KAN_-_Kolmogorov-Arnold_Networks.md","slug":"KAN - Kolmogorov-Arnold Networks","pdf":"KAN - Kolmogorov-Arnold Networks.pdf","ref_equations":37,"inline_structural":545,"notation_symbols":70},{"file":"Deep_Hedging.md","slug":"Deep Hedging","pdf":"Deep Hedging.pdf","ref_equations":39,"inline_structural":447,"notation_symbols":86},{"file":"Transformers_are_SSMs_-_Generalized_Models_and_Efficient_Algorithms_Through_Structured_State_Space_Duality.md","slug":"Transformers are SSMs - Generalized Models and Efficient Algorithms Through Structured State Space Duality","pdf":"Transformers are SSMs - Generalized Models and Efficient Algorithms Through Structured State Space Duality.pdf","ref_equations":54,"inline_structural":405,"notation_symbols":117},{"file":"Operator_Deep_Smoothing_for_Implied_Volatility.md","slug":"Operator Deep Smoothing for Implied Volatility","pdf":"Operator Deep Smoothing for Implied Volatility.pdf","ref_equations":47,"inline_structural":261,"notation_symbols":98},{"file":"Neural_Controlled_Differential_Equations_for_Irregular_Time_Series.md","slug":"Neural Controlled Differential Equations for Irregular Time Series","pdf":"Neural Controlled Differential Equations for Irregular Time Series.pdf","ref_equations":55,"inline_structural":248,"notation_symbols":67},{"file":"The_Neural_Hawkes_Process_-_A_Neurally_Self-Modulating_Multivariate_Point_Process.md","slug":"The Neural Hawkes Process - A Neurally Self-Modulating Multivariate Point Process","pdf":"The Neural Hawkes Process - A Neurally Self-Modulating Multivariate Point Process.pdf","ref_equations":15,"inline_structural":285,"notation_symbols":64},{"file":"Signature-Informed_Transformer_for_Asset_Allocation.md","slug":"Signature-Informed Transformer for Asset Allocation","pdf":"Signature-Informed Transformer for Asset Allocation.pdf","ref_equations":40,"inline_structural":250,"notation_symbols":117},{"file":"Simplified_State_Space_Layers_for_Sequence_Modeling.md","slug":"Simplified State Space Layers for Sequence Modeling","pdf":"Simplified State Space Layers for Sequence Modeling.pdf","ref_equations":33,"inline_structural":237,"notation_symbols":65},{"file":"Efficiently_Modeling_Long_Sequences_with_Structured_State_Spaces.md","slug":"Efficiently Modeling Long Sequences with Structured State Spaces","pdf":"Efficiently Modeling Long Sequences with Structured State Spaces.pdf","ref_equations":50,"inline_structural":205,"notation_symbols":64},{"file":"Optimal_execution_with_rough_path_signatures.md","slug":"Optimal execution with rough path signatures","pdf":"Optimal execution with rough path signatures.pdf","ref_equations":14,"inline_structural":228,"notation_symbols":55},{"file":"Pretrained_Time-Series_Foundation_Models_for_Financial_Return_Forecasting.md","slug":"Pretrained Time-Series Foundation Models for Financial Return Forecasting","pdf":"Pretrained Time-Series Foundation Models for Financial Return Forecasting.pdf","ref_equations":62,"inline_structural":178,"notation_symbols":141},{"file":"Re_Visiting_Time_Series_Foundation_Models_in_Finance.md","slug":"Re(Visiting) Time Series Foundation Models in Finance","pdf":"Re(Visiting) Time Series Foundation Models in Finance.pdf","ref_equations":34,"inline_structural":180,"notation_symbols":79},{"file":"Kronos_-_A_Foundation_Model_for_the_Language_of_Financial_Markets.md","slug":"Kronos - A Foundation Model for the Language of Financial Markets","pdf":"Kronos - A Foundation Model for the Language of Financial Markets.pdf","ref_equations":14,"inline_structural":151,"notation_symbols":80},{"file":"Transformer_Hawkes_Process.md","slug":"Transformer Hawkes Process","pdf":"Transformer Hawkes Process.pdf","ref_equations":23,"inline_structural":129,"notation_symbols":76},{"file":"Reinforcement_Learning_in_Non-Markov_Market-Making.md","slug":"Reinforcement Learning in Non-Markov Market-Making","pdf":"Reinforcement Learning in Non-Markov Market-Making.pdf","ref_equations":36,"inline_structural":115,"notation_symbols":42},{"file":"Chronos-_Learning_the_Language_of_Time_Series.md","slug":"Chronos- Learning the Language of Time Series","pdf":"Chronos- Learning the Language of Time Series.pdf","ref_equations":4,"inline_structural":130,"notation_symbols":65},{"file":"CARD_-_Channel_Aligned_Robust_Blend_Transformer_for_Time_Series_Forecasting.md","slug":"CARD - Channel Aligned Robust Blend Transformer for Time Series Forecasting","pdf":"CARD - Channel Aligned Robust Blend Transformer for Time Series Forecasting.pdf","ref_equations":15,"inline_structural":117,"notation_symbols":49},{"file":"CausalStock_-_Deep_End-to-end_Causal_Discovery_for_News-driven_Stock_Movement_Prediction.md","slug":"CausalStock - Deep End-to-end Causal Discovery for News-driven Stock Movement Prediction","pdf":"CausalStock - Deep End-to-end Causal Discovery for News-driven Stock Movement Prediction.pdf","ref_equations":18,"inline_structural":102,"notation_symbols":51},{"file":"MarS_-_a_Financial_Market_Simulation_Engine_Powered_by_Generative_Foundation_Model.md","slug":"MarS - a Financial Market Simulation Engine Powered by Generative Foundation Model","pdf":"MarS - a Financial Market Simulation Engine Powered by Generative Foundation Model.pdf","ref_equations":8,"inline_structural":97,"notation_symbols":41},{"file":"ChronosX_-_Adapting_Pretrained_Time_Series_Models_with_Exogenous_Variables.md","slug":"ChronosX - Adapting Pretrained Time Series Models with Exogenous Variables","pdf":"ChronosX - Adapting Pretrained Time Series Models with Exogenous Variables.pdf","ref_equations":19,"inline_structural":79,"notation_symbols":56},{"file":"SigKAN_-_Signature-Weighted_Kolmogorov-Arnold_Networks_for_Time_Series.md","slug":"SigKAN - Signature-Weighted Kolmogorov-Arnold Networks for Time Series","pdf":"SigKAN - Signature-Weighted Kolmogorov-Arnold Networks for Time Series.pdf","ref_equations":17,"inline_structural":76,"notation_symbols":40},{"file":"The_Inference-Compute_Frontier_and_a_Latency-Efficient_Architecture_for_Limit_Order_Book_Prediction.md","slug":"The Inference-Compute Frontier and a Latency-Efficient Architecture for Limit Order Book Prediction","pdf":"The Inference-Compute Frontier and a Latency-Efficient Architecture for Limit Order Book Prediction.pdf","ref_equations":0,"inline_structural":92,"notation_symbols":29},{"file":"Deep_Reinforcement_Learning_for_Market_Making_Under_a_Hawkes_Process-Based_Limit_Order_Book_Model.md","slug":"Deep Reinforcement Learning for Market Making Under a Hawkes Process-Based Limit Order Book Model","pdf":"Deep Reinforcement Learning for Market Making Under a Hawkes Process-Based Limit Order Book Model.pdf","ref_equations":15,"inline_structural":73,"notation_symbols":38},{"file":"Unified_Training_of_Universal_Time_Series_Forecasting_Transformers.md","slug":"Unified Training of Universal Time Series Forecasting Transformers","pdf":"Unified Training of Universal Time Series Forecasting Transformers.pdf","ref_equations":5,"inline_structural":83,"notation_symbols":64},{"file":"CAMEF_-_Causal-Augmented_Multi-Modality_Event-Driven_Financial_Forecasting_by_Integrating_Time_Series_Patterns_and_Salie.md","slug":"CAMEF - Causal-Augmented Multi-Modality Event-Driven Financial Forecasting by Integrating Time Series Patterns and Salient Macroeconomic Announcements","pdf":"CAMEF - Causal-Augmented Multi-Modality Event-Driven Financial Forecasting by Integrating Time Series Patterns and Salient Macroeconomic Announcements.pdf","ref_equations":18,"inline_structural":69,"notation_symbols":59},{"file":"Extracting_information_from_the_signature_of_a_financial_data_stream.md","slug":"Extracting information from the signature of a financial data stream","pdf":"Extracting information from the signature of a financial data stream.pdf","ref_equations":3,"inline_structural":79,"notation_symbols":24},{"file":"DiffVolume_-_Diffusion_Models_for_Volume_Generation_in_Limit_Order_Books.md","slug":"DiffVolume - Diffusion Models for Volume Generation in Limit Order Books","pdf":"DiffVolume - Diffusion Models for Volume Generation in Limit Order Books.pdf","ref_equations":11,"inline_structural":70,"notation_symbols":43},{"file":"FinCast_-_A_Foundation_Model_for_Financial_Time-Series_Forecasting.md","slug":"FinCast - A Foundation Model for Financial Time-Series Forecasting","pdf":"FinCast - A Foundation Model for Financial Time-Series Forecasting.pdf","ref_equations":22,"inline_structural":52,"notation_symbols":61},{"file":"xLSTM-Mixer_-_Multivariate_Time_Series_Forecasting_by_Mixing_via_Scalar_Memories.md","slug":"xLSTM-Mixer - Multivariate Time Series Forecasting by Mixing via Scalar Memories","pdf":"xLSTM-Mixer - Multivariate Time Series Forecasting by Mixing via Scalar Memories.pdf","ref_equations":3,"inline_structural":66,"notation_symbols":56},{"file":"Time-MoE_-_Billion-Scale_Time_Series_Foundation_Models_with_Mixture_of_Experts.md","slug":"Time-MoE - Billion-Scale Time Series Foundation Models with Mixture of Experts","pdf":"Time-MoE - Billion-Scale Time Series Foundation Models with Mixture of Experts.pdf","ref_equations":8,"inline_structural":56,"notation_symbols":50},{"file":"Generative_AI_for_End-to-End_Limit_Order_Book_Modelling_-_A_Token-Level_Autoregressive_Generative_Model_of_Message_Flow_.md","slug":"Generative AI for End-to-End Limit Order Book Modelling - A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network","pdf":"Generative AI for End-to-End Limit Order Book Modelling - A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network.pdf","ref_equations":6,"inline_structural":50,"notation_symbols":32},{"file":"iTransformer_-_Inverted_Transformers_Are_Effective_for_Time_Series_Forecasting.md","slug":"iTransformer - Inverted Transformers Are Effective for Time Series Forecasting","pdf":"iTransformer - Inverted Transformers Are Effective for Time Series Forecasting.pdf","ref_equations":3,"inline_structural":53,"notation_symbols":38},{"file":"ByteGen_-_A_Tokenizer-Free_Generative_Model_for_Orderbook_Events_in_Byte_Space.md","slug":"ByteGen - A Tokenizer-Free Generative Model for Orderbook Events in Byte Space","pdf":"ByteGen - A Tokenizer-Free Generative Model for Orderbook Events in Byte Space.pdf","ref_equations":16,"inline_structural":37,"notation_symbols":40},{"file":"LOB-Bench_-_Benchmarking_Generative_AI_for_Finance_--_an_Application_to_Limit_Order_Book_Data.md","slug":"LOB-Bench - Benchmarking Generative AI for Finance -- an Application to Limit Order Book Data","pdf":"LOB-Bench - Benchmarking Generative AI for Finance -- an Application to Limit Order Book Data.pdf","ref_equations":7,"inline_structural":45,"notation_symbols":28},{"file":"VisionTS_-_Visual_Masked_Autoencoders_Are_Free-Lunch_Zero-Shot_Time_Series_Forecasters.md","slug":"VisionTS - Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters","pdf":"VisionTS - Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters.pdf","ref_equations":1,"inline_structural":51,"notation_symbols":21},{"file":"TimeXer-_Empowering_Transformers_for_Time_Series_Forecasting_with_Exogenous_Variables.md","slug":"TimeXer- Empowering Transformers for Time Series Forecasting with Exogenous Variables","pdf":"TimeXer- Empowering Transformers for Time Series Forecasting with Exogenous Variables.pdf","ref_equations":8,"inline_structural":41,"notation_symbols":35},{"file":"Lag-Llama_-_Towards_Foundation_Models_for_Probabilistic_Time_Series_Forecasting.md","slug":"Lag-Llama - Towards Foundation Models for Probabilistic Time Series Forecasting","pdf":"Lag-Llama - Towards Foundation Models for Probabilistic Time Series Forecasting.pdf","ref_equations":5,"inline_structural":39,"notation_symbols":33},{"file":"Are_KANs_Effective_for_Multivariate_Time_Series_Forecasting.md","slug":"Are KANs Effective for Multivariate Time Series Forecasting?","pdf":"Are KANs Effective for Multivariate Time Series Forecasting?.pdf","ref_equations":11,"inline_structural":31,"notation_symbols":50},{"file":"Multimodal_Language_Models_with_Modality-Specific_Experts_for_Financial_Forecasting_from_Interleaved_Sequences_of_Text_a.md","slug":"Multimodal Language Models with Modality-Specific Experts for Financial Forecasting from Interleaved Sequences of Text and Time Series","pdf":"Multimodal Language Models with Modality-Specific Experts for Financial Forecasting from Interleaved Sequences of Text and Time Series.pdf","ref_equations":5,"inline_structural":35,"notation_symbols":24},{"file":"Moirai-MoE_-_Empowering_Time_Series_Foundation_Models_with_Sparse_Mixture_of_Experts.md","slug":"Moirai-MoE - Empowering Time Series Foundation Models with Sparse Mixture of Experts","pdf":"Moirai-MoE - Empowering Time Series Foundation Models with Sparse Mixture of Experts.pdf","ref_equations":6,"inline_structural":23,"notation_symbols":42},{"file":"JaxMARL-HFT_-_GPU-Accelerated_Large-Scale_Multi-Agent_Reinforcement_Learning_for_High-Frequency_Trading.md","slug":"JaxMARL-HFT - GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading","pdf":"JaxMARL-HFT - GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading.pdf","ref_equations":4,"inline_structural":8,"notation_symbols":9},{"file":"FinMultiTime_-_A_Four-Modal_Bilingual_Dataset_for_Financial_Time-Series_Analysis.md","slug":"FinMultiTime - A Four-Modal Bilingual Dataset for Financial Time-Series Analysis","pdf":"FinMultiTime - A Four-Modal Bilingual Dataset for Financial Time-Series Analysis.pdf","ref_equations":0,"inline_structural":8,"notation_symbols":1},{"file":"TimeGPT-1.md","slug":"TimeGPT-1","pdf":"TimeGPT-1.pdf","ref_equations":2,"inline_structural":5,"notation_symbols":13},{"file":"Rethinking_Evaluation_in_the_Era_of_Time_Series_Foundation_Models-_Un_known_Information_Leakage_Challenges.md","slug":"Rethinking Evaluation in the Era of Time Series Foundation Models- (Un)known Information Leakage Challenges","pdf":"Rethinking Evaluation in the Era of Time Series Foundation Models- (Un)known Information Leakage Challenges.pdf","ref_equations":0,"inline_structural":6,"notation_symbols":2}]}
const papers = input.papers
const mdRoot = input.md_root
if (!Array.isArray(papers)) throw new Error('args.papers must be an array')

log(`${papers.length} papers · ${papers.reduce((a, p) => a + p.ref_equations, 0)} display equations · ` +
    `${papers.reduce((a, p) => a + p.inline_structural, 0)} structural inline spans · md_root=${mdRoot}`)

const results = await pipeline(
  papers,
  (p) => agent(validatorPrompt(p, mdRoot), {
    label: `validate:${p.slug.slice(0, 32)}`,
    phase: 'Validate',
    schema: VALIDATOR_SCHEMA,
  }),
  async (val, p) => {
    if (!val) return { paper: p, validation: null, verification: null }
    const f = val.findings || []
    if (f.length === 0) return { paper: p, validation: val, verification: { verdicts: [] } }
    const ver = await agent(verifierPrompt(p, f, mdRoot), {
      label: `verify:${p.slug.slice(0, 34)} (${f.length})`,
      phase: 'Verify',
      schema: VERIFIER_SCHEMA,
    })
    return { paper: p, validation: val, verification: ver }
  }
)

const ok = results.filter(Boolean)
log(`done: ${ok.length}/${papers.length}, ${ok.reduce((a, r) => a + (r.validation?.findings?.length || 0), 0)} findings, ` +
    `${ok.reduce((a, r) => a + (r.validation?.notes?.length || 0), 0)} notes`)
return ok
