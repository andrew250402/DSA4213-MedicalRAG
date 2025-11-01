# CancerGov QA Dataset

This folder contains XML files with cancer-related question-answer pairs sourced from the National Cancer Institute (NCI) website.

## URL Corrections

Several head and neck cancer treatment documents had incorrect URLs in their original XML files. The URLs were missing `/adult` in the path. The following documents have been corrected:

| Document ID | Cancer Type                                         | Corrected URL                                                                                     |
| ----------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 0000006_2   | Brain Tumors in Children                            | <https://www.cancer.gov/types/brain/hp/child-brain-treatment-pdq>                               |
| 0000007_3   | Liver Cancer in Children                            | <https://www.cancer.gov/types/liver/hp/child-liver-treatment-pdq>                               |
| 0000024_1   | Hypopharyngeal Cancer                               | <https://www.cancer.gov/types/head-and-neck/patient/adult/hypopharyngeal-treatment-pdq>           |
| 0000024_2   | Laryngeal Cancer                                    | <https://www.cancer.gov/types/head-and-neck/patient/adult/laryngeal-treatment-pdq>                |
| 0000024_3   | Lip and Oral Cavity Cancer                          | <https://www.cancer.gov/types/head-and-neck/patient/adult/lip-mouth-treatment-pdq>                |
| 0000024_4   | Metastatic Squamous Neck Cancer with Occult Primary | <https://www.cancer.gov/types/head-and-neck/patient/adult/metastatic-squamous-neck-treatment-pdq> |
| 0000024_5   | Nasopharyngeal Cancer                               | <https://www.cancer.gov/types/head-and-neck/patient/adult/nasopharyngeal-treatment-pdq>           |
| 0000024_6   | Oropharyngeal Cancer                                | <https://www.cancer.gov/types/head-and-neck/patient/adult/oropharyngeal-treatment-pdq>            |
| 0000024_7   | Paranasal Sinus and Nasal Cavity Cancer             | <https://www.cancer.gov/types/head-and-neck/patient/adult/paranasal-sinus-treatment-pdq>          |
| 0000024_8   | Salivary Gland Cancer                               | <https://www.cancer.gov/types/head-and-neck/patient/adult/salivary-gland-treatment-pdq>           |

### Pattern

The correction involved adding `/adult` to the URL path.

- **Old pattern:** `https://www.cancer.gov/types/head-and-neck/patient/{cancer-type}-treatment-pdq`
- **New pattern:** `https://www.cancer.gov/types/head-and-neck/patient/adult/{cancer-type}-treatment-pdq`

Or add `/hp/` for documents related to children.

- **Old pattern:** `https://www.cancer.gov/types/{cancer-type}/patient/{child-cancer-type}-treatment-pdq`
- **New pattern:** `https://www.cancer.gov/types/{cancer-type}/hp/{child-cancer-type}-treatment-pdq`

## File Structure

Each XML file contains:

- Document metadata (id, source, url)
- Focus area (cancer type)
- UMLS annotations (CUIs, Semantic Types)
- Question-Answer pairs covering:
  - General information
  - Risk factors
  - Symptoms
  - Diagnosis methods
  - Staging
  - Treatment options
  - Clinical trials

## Data Source

All documents are sourced from the National Cancer Institute's PDQ® (Physician Data Query) Cancer Information Summaries - Patient Version.
