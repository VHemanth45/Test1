# HumanAI
---
## Project Name
Optical Character Recognition of printed sources

## Description
Build a model based on convolutional-recurrent, transformer, or self-supervised
architectures for optically recognizing the text of each data source. Your model should be
able to detect the main text in each page, while disregarding other embellishments.
Integrate an LLM or VLM model as a late-stage step to the OCR process (such as cleaning
up the OCR output). Pick the most appropriate approach and discuss your strategy

## Project Summary
I built The RenAIssance OCR Pipeline, an end-to-end optical character recognition system for historical printed documents containing complex layouts, decorative elements, and archaic spellings. The pipeline begins by isolating the main text regions using a fine-tuned YOLO-based object detection model, allowing the system to bypass headers, marginalia, and ornamental content that typically interfere with OCR performance. I then segmented the detected regions into line-level crops and aligned them with ground-truth transcripts to create a robust supervised training dataset.

To identify the most effective OCR strategy, I implemented and evaluated multiple architectures against a Tesseract baseline:
1. Transformer-based OCR fine-tuning using TrOCR
2. Convolutional-Recurrent Neural Network (CRNN) training using Kraken
3. LLM-based post-correction to refine OCR outputs while preserving historical spelling, punctuation, and document-specific artifacts
By integrating these components into a complete OCR workflow, I developed a final hybrid system (CRNN + LLM) that achieved an 80% reduction in Character Error Rate (CER) compared with the baseline.

## Pipeline Followed
1. **Document Processing**
   - Convert source PDFs into page images.
   - Detect and crop text regions/lines for OCR-ready inputs.

2. **Text-Image Alignment & Dataset Curation**
   - Align line crops with ground-truth transcriptions.
   - Correct misalignments and build clean train/validation data.

<img width="1189" height="193" alt="image" src="https://github.com/user-attachments/assets/f0060de0-0e7e-4dc1-8003-bd1d94346104" />
<img width="1189" height="159" alt="image" src="https://github.com/user-attachments/assets/446424fe-6235-44d8-8cfd-c0154c4d98de" />
<img width="1489" height="881" alt="image" src="https://github.com/user-attachments/assets/d0656131-e1a2-44b1-9b93-d7157c6211dc" />

3. **Baseline OCR Evaluation (Tesseract)**
   - Establish baseline performance on the locked validation set.

4. **Transformer OCR (TrOCR Fine-tuning)**
   - Fine-tune TrOCR and evaluate with CER, WER, and chrF.

5. **Convolutional-Recurrent OCR (Kraken CRNN)**
   - Train/evaluate CRNN as the convolutional-recurrent architecture.
   - Compare against baseline and TrOCR on the same validation split.

6. **LLM Post-Correction**
   - Apply contextual correction over OCR output.
   - Re-evaluate to quantify final gains in CER/WER/chrF.

7. **Final Comparison & Analysis**
   - Report stage-wise improvements and select the best pipeline.

### Metrics Comparison Table
| Stage                           | CER              | WER              | chrF             | Samples |
| ------------------------------- | ---------------- | ---------------- | ---------------- | ------- |
| Tesseract Baseline              | 0.3553           | 0.7713           | 0.4850           | 296     |
| TrOCR Fine-tuned                | 0.1750           | 0.4657           | 0.6507           | 296     |
| CRNN (Kraken)                   | 0.0928           | 0.3335           | 0.7837           | 296     |
| **CRNN + LLM Correction** | **0.0724** | **0.2363** | **0.8493** | 296     |

### Overall Improvement
| Journey                                   | CER                        | Relative Improvement |
| ----------------------------------------- | -------------------------- | -------------------- |
| Tesseract → TrOCR                        | 0.3553 → 0.1750           | 51%                  |
| TrOCR → CRNN                             | 0.1750 → 0.0928           | 47%                  |
| CRNN → CRNN + LLM Correction             | 0.0928 → 0.0724           | 22%                  |
| **Tesseract → Final (CRNN + LLM)** | **0.3553 → 0.0724** | **80%**        |

<img width="866" height="820" alt="image" src="https://github.com/user-attachments/assets/ce36e1f4-f236-45f4-8570-0b298b239001" />


| GroundTruth                                                  | CRNN Prediction                                           | LLM Corrected                                              |
| ------------------------------------------------------------ | --------------------------------------------------------- | ---------------------------------------------------------- |
| yor parte.                                                   | yor parte, a.                                             | yor parte, a.                                              |
| Pero este discurso repugna a la jurisprudencia, y a la       | ero este discurso epugna a la jurisprudencia, y a la      | Pero este discurso repugna a la jurisprudencia, y a la     |
| resolucion mas assentada que ay en este Reyno.               | resolucion mas assentada que ay en este Rteyno, -         | resolucion mas assentada que hay en este Reyno, -          |
| Lo primero, que la viuda tiene el primer lugar en la         | Lo primero, que la viuda tiene el primer lugar en la      | Lo primero, que la viuda tiene el primer lugar en la       |
| acusacion de la muerte del marido, y demas agravios          | acusacion de la muerte del marido, y demas agravios       | acusacion de la muerte del marido, y demas agravios        |
| hechos a su casa, sin que los hijos (quando huvieran acu-    | hechos a sur casa, sin que los hijos squado huvieran acu- | hechos a su casa, sin que los hijos squado hubieran acu-   |
| sado) puedan concurrir con ella, ni ser parte para remi-     | sado puedan concurtir con ella, niser parte para temi-    | sado puedan concurrir con ella, ni ser parte para temi-    |
| tir la muerte, como lo assientan los Autores q adelante      | tir la muerre, como lo assientan los Autores qadelante    | tir la muerte, como lo assientan los Autores que adelante  |
| se citaran, fundadas en la razon de que el derecho de        | se citaran, sundados en la razon de que el derecho de     | se citaran, fundados en la razon de que el derecho de      |
| acusar la muerte, non est quid haereditarium, I. pro haere   | acusan la muerte, non est quid hareditarium. Lpro here    | acusar la muerte, non est quid hereditarium. L. pro here   |
| de, §, sin, ff, de acquiren. haered. & sic, la muger aunque | deS. sin si. de aequitena hared. e sie, la muger aunque   | de §. sin si. de aequitate hered. et sic, la mujer aunque |
| no sea heredera de su marido, se prefiere en la acusacion    | no sea heredeta de su marido, se prefiere en la acusacio  | no sea heredera de su marido, se prefiere en la acusacion  |
| de su muerte a los hijos, y a todos los demas consangui-     | de sus muerte a los hijos, y a todos los demas cosangui-  | de su muerte a los hijos, y a todos los demas cosangui-    |
| neos porque como por el matrimonio se hizieron am-           | neos: porque como por el matrimonio se hizieron an-       | neos: porque como por el matrimonio se hicieron an-        |
| bos una misma carne y sangre, no ay persona mas con-         | bos vnamisma carne y sangte, no ay persona mas con-       | bos una misma carne y sangre, no hay persona mas con-      |
| junta que ella, & sunevnares, & vnum fuppofitum, c.          | junta que ella, de funt una rea, es unun suppositum, e    | junta que ella, de fuit una res, es unum suppositum, et    |

## Notebook
All preprocessing steps, intermediate outputs, and result visualizations are documented in the notebook [RenAIssance_Showcase.ipynb](RenAIssance_Showcase.ipynb), while the core implementation files and complete pipeline code are available in the main repository.
