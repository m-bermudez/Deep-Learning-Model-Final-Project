
import nltk
import textwrap
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

# Download sentence tokenizer
nltk.download('punkt')

# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Reference phrases for diabetes-related concepts
diabetes_refs = [
    # Core diabetes conditions
    "diabetes", "diabetic", "DM1", "DM Type 1", "DM Type I", "type 1 diabetes", "type 2 diabetes",
    "diabetes diagnosis", "DKA", "ketoacidosis", "DKA history", "DKA episode", "DKA resolved", "DKA management",

    # Symptoms and clinical context
    "hyperglycemia", "hypoglycemia", "hyperglycemia treatment",
    "polyuria", "polydipsia", "polyphagia", "HPI", "blood sugar", "blood sugar control",

    # Lab measurements and entries
    "glucose", "glucose-", "FS=", "FS:", "FS-", "FS(", "HbA1C", "a1c", "A1C test results",
    "ketone", "anion gap", "bicarbonate", "blood glucose", "blood sugars",

    # Medications and treatments
    "insulin", "insulin therapy", "Lantus", "Novalog", "Humalog",
    "sliding scale", "sliding scale insulin", "subcutaneously", "insulin drip", "insulin gtt", "insulin sliding scale", "amp D50",

    # Nutrition & diet
    "carbohydrate", "diabetic diet", "glucose control"
]

ref_embeds = model.encode(diabetes_refs, convert_to_tensor=True)

# Long clinical note (replace or read from file if needed)
long_note = """Admission Date:  [**2125-7-10**]              Discharge Date:   [**2125-7-11**]  Date of Birth:
[**2105-5-5**]             Sex:   F  Service: MED  Allergies: Patient recorded as having No Known
Allergies to Drugs  Attending:[**First Name3 (LF) 2641**] Chief Complaint: Diabetic Ketoacidosis
Major Surgical or Invasive Procedure: None  History of Present Illness: Pt is a 20 yo female with
Type I Diabetes (Dxd in 11/00) who was found to be in DKA after an ambulance was called s/p
sustaining a mechanical fall on [**2125-7-9**] ~4 pm. Pt was at work folding laundry when she felt
her "back lock up", fell to the floor, hit her head, and then went to her house. She threw up after
hitting her head and vomited 3-4 times that day. [**Name (NI) **] mother was concerned about the
fall and called an ambulance. In ED FS=497, K = 6.8, Bicarbonate was 10, and AG was 33. She received
7 u insulin subcutaneously and was started on Insulin gtt (7 units/hour) and received ~5 L NS.  Back
pain has been constant since getting hit in a MVA in [**Month (only) **] this year. The pain is
sharp, and is localized to the right lumbo-sacral region. She believes that this is why her back
locked up and it is not uncommon.  The morning of the fall patient said her FS was 138 and she had a
slushy right after, though she did not know the carbohydrate equivalents.  On ROS no diarrhea.  No
sick contacts. [**Name (NI) **] polyuria/ polydipsia/ polyphagia. No H/A or visual changes. No
tremors. Does report "a funny discharge" from her vagina which is yellow and started the day of
admission. No dyspareunia. Of note, patient had gonorrhea in [**Month (only) 958**], when she was
admitted to [**Hospital3 **] with DKA and said that that was the tipoff that time.  She says that
she is sexually active with her partner of 6 years and that they use condoms "usually." In [**Month
(only) 958**], patient's partner was also treated for gonorrhea. There were a few months when they
were not dating and he had sexual relations with someone else. Otherwise, the patient says that they
are both monogomous.  Patient was diagnosed with DM I in 11/00 shortly after suffering a
miscarriage. She presented with H/A and went to [**Hospital1 2177**] and was diagnosed. She has been
in DKA ~5 times since that time, most recently in [**Month (only) 958**] (as above). She reports
having good recent control since starting at [**Last Name (un) **] earlier this year. She takes
Novalog 1u/10 g of carbohydrates and takes 35 units of lantus at night. She takes her FS ~3-4 times
per day and reports a usual range of 65-225.  Past Medical History: 1.Diabetes Type I as above.
2.Hyperlipidemia 3. S/P MVA [**5-4**]-Right lower back pain since then. + Back spasms treated with
tylenol. 4. Goiter 5. Depression   Social History: Patient started work as a personalized care
attendant on day of admission. Completed high school in [**2122**]. She has a two-year-old son with
her current partner. Quit smoking two years ago. [**6-7**] cigarettes per week for 3 years. No EtOH.
No marijuana, cocaine, heroin or other recreational drugs.  Family History: GM with Type I diabetes.
Otherwise non-contributory.  Physical Exam: On admission to medicine floor from MICU:  VS: T: 98.6;
BP: 116/55, P: 75; RR:15; O2: 99%; I/O 24 hour:[**Numeric Identifier **]/4775 FS: 0300 (214) 0400
(288) 1000 (92) 1300 (352) Gen: Laying in bed in NAD HEENT: PERRL, EOMI, OP clear no exudate,
tongue-ring in place, MMM Neck: No JVD, No LAD. Painful to palpation left anterior cervical area.
CV:RRR s1s2. No M/R/G. Lungs: CTA b/l. good air entry. Abd: + BS, soft, NT, ND. Ext: 2+ DP. No
C/C/E. No tremors. Back: No pain to deep palpation. No CVA tenderness. Neuro: Reflexes 3+ b/l
patellar, biceps.   Pertinent Results: Labs on Admission:  [**2125-7-9**] 08:52PM URINE  RBC-0-2
WBC-0-2 BACTERIA-RARE YEAST-NONE EPI-0-2 BLOOD-MOD NITRITE-NEG PROTEIN-TR GLUCOSE-1000 KETONE-150
BILIRUBIN-NEG UROBILNGN-NEG PH-5.0 LEUK-NEG COLOR-Straw APPEAR-Clear SP [**Last Name (un)
155**]-1.030  [**2125-7-9**]   WBC-16.0* RBC-5.04 HGB-14.5 HCT-46.3 MCV-92 MCH-28.8 MCHC-31.3
RDW-13.3 PLT COUNT-232 HYPOCHROM-2+ NEUTS-86.0* LYMPHS-11.9* MONOS-1.4* EOS-0.2 BASOS-0.5
ALT(SGPT)-17 AST(SGOT)-40 ALK PHOS-116 AMYLASE-76 TOT BILI-0.4 GLUCOSE-489* UREA N-24* CREAT-1.2*
SODIUM-132* POTASSIUM-6.8* CHLORIDE-94* TOTAL CO2-5* ANION GAP-40* [**2125-7-9**] 10:58PM
GLUCOSE-457* NA+-136 K+-6.0* CL--99* TCO2-10* [**2125-7-10**] 01:00AM   GLUCOSE-202* UREA N-22*
CREAT-1.1 SODIUM-138 POTASSIUM-6.8* CHLORIDE-106 TOTAL CO2-7* ANION GAP-32*  Chem 7s- [**2125-7-9**]
10:40PM Glucose-489* UreaN-24* Creat-1.2* Na-132* K-6.8* Cl-94* HCO3-5* [**2125-7-10**] 01:00AM
Glucose-202* UreaN-22* Creat-1.1 Na-138 K-6.8* Cl-106 HCO3-7* [**2125-7-10**] 04:00AM Glucose-219*
UreaN-16 Creat-0.9 Na-138 K-3.9 Cl-112* HCO3-7* [**2125-7-10**] 08:16AM Glucose-161* UreaN-10
Creat-0.7 Na-136 K-3.6 Cl-112* HCO3-12* [**2125-7-10**] 02:30PM Glucose-130* UreaN-8 Creat-1.0
Na-136 K-3.7 Cl-113* HCO3-16* [**2125-7-10**] 06:00PM Glucose-79 UreaN-9 Creat-0.6 Na-138 K-3.5
Cl-116* HCO3-15* [**2125-7-11**] 06:09AM Glucose-164* UreaN-6 Creat-0.6 Na-137 K-3.4 Cl-111*
HCO3-17*  [**2125-7-11**] 05:20PM Glucose-120* UreaN-10 Creat-0.8 Na-138 K-3.6 Cl-104 HCO3-22
[**2125-7-11**] 06:09AM BLOOD WBC-5.5# RBC-3.90* Hgb-11.5* Hct-34.3*# MCV-88 MCH-29.5 MCHC-33.7
RDW-13.6 Plt Ct-89*#  Last day of hospitalization  [**2125-7-11**]  Glucose-120* UreaN-10 Creat-0.8
Na-138 K-3.6 Cl-104 HCO3-22 [**2125-7-11**]  ALT-17 AST-25 AlkPhos-69 Amylase-86 TotBili-0.6
Calcium-8.5 Phos-1.9* Mg-1.8 WBC-5.5# RBC-3.90* Hgb-11.5* Hct-34.3*# MCV-88 MCH-29.5 MCHC-33.7
RDW-13.6 Plt Ct-89*#  EKG: [**2125-7-9**]- Sinus tachycardia at 105 bpm. Irregular rhythm with
premature atrial beats. T wave inversions in V1 and V2. [**2125-7-10**]-Sinus rhythm at 95. Normal
rate. Small ST depression in V1. Less prominent than previous  EKG.   Brief Hospital Course: *** Pt
left AMA on the night of [**2125-7-11**] secondary to childcare issues. Attending and house staff
both went over the risks of leaving AMA, including but not limited to dehydration, hyperglycemia,
diabetic ketoacidosis, coma, and death. Also, the patient was made aware that her plateletes had
decreased dramatically and leaving against medical advice could lead to increased risk of
bleeding.***  1. DKA   Patient was continued on insulin drip at 7cc/hour upon arrival to the MICU.
On [**7-11**] ~12:30 am insulin drip was d/cd and patient had hypoglycemia to 52 (received amp D50).
Anion gap slowly closed by the morning of [**2125-7-11**], however with a bicarbonate of 17.
Patient was transferred to the floor and a [**Last Name (un) **] fellow consulted on the case. The
humalog insulin sliding scale was changed to 4 units Humalog standing before each meal and 1 unit of
insulin for every 50 of glucose greater than 200. We were also going to continue the patient on
Lantus 30 units qhs. Patient's blood sugars were in upper 100s-200s on day of discharge with blood
sugar going up to above 300 at times.  HgA1C was tested and found to be 11.6. Therefore usual
glucose is usually > 300 and indicates that patient is poorly controlled.  2. Cause of DKA   Pt with
vomiting upon arrival. Could have been from DKA itself. No history in days prior to presentation of
vomiting. On transfer to the medicine floor from the MICU, the N/V had resolved. Another possible
etiology could be a vaginal infection as patient says that she has a yellow discharge which is
similar to when she had gonorrhea in [**Month (only) 958**]. The plan was to perform a gynecological
exam. However, the patient left before being able to do so.  3. Backpain   Ms. [**Known lastname **]
has had backpain since being in an MVA in [**Month (only) **]. She takes tylenol for the pain and
this was continued as an inpatient.  4. F/E/N   On the day of leaving the hospital, Ms. [**Known
lastname **] was tolerating PO and was on a diabetic carbohydrate consistent diet.  5. Access: PVLs
6. Code Status: Full Code while in the hospital   Medications on Admission: 1. Lantus 35 units qhs
2. Novalog sliding scale-1 unit for every 10 grams of carbohydrate 3. Lipitor 20 mg once a day 4.
Trazadone 100 mg qhs   Discharge Medications: Patient left AMA. She will continue her home
medications.  Discharge Disposition: Home  Discharge Diagnosis: Diabetic Ketoacidosis
Hyperlipidemia Depression Lower back pain  Discharge Condition: Fair  Discharge Instructions:
Patient left AMA.  Followup Instructions: Patient left AMA. She was urged to follow-up with her
[**Last Name (un) **] physician the day after discharge for an appointment within a few days. Pt
also said that she had an appointment with her PCP two days after leaving the hospital."""  # Truncated for brevity

# Tokenize into sentences
sentences = sent_tokenize(long_note)

# Filter by diabetes-related terms
diabetes_terms = [
    # Core diabetes conditions
    "diabetes", "diabetic", "DM1", "DM Type 1", "DM Type I", "type 1 diabetes", "type 2 diabetes",
    "diabetes diagnosis", "DKA", "ketoacidosis", "DKA history", "DKA episode", "DKA resolved", "DKA management",

    # Symptoms and clinical context
    "hyperglycemia", "hypoglycemia", "hyperglycemia treatment",
    "polyuria", "polydipsia", "polyphagia", "HPI", "blood sugar", "blood sugar control",

    # Lab measurements and entries
    "glucose", "glucose-", "FS=", "FS:", "FS-", "FS(", "HbA1C", "a1c", "A1C test results",
    "ketone", "anion gap", "bicarbonate", "blood glucose", "blood sugars",

    # Medications and treatments
    "insulin", "insulin therapy", "Lantus", "Novalog", "Humalog",
    "sliding scale", "sliding scale insulin", "subcutaneously", "insulin drip", "insulin gtt", "insulin sliding scale", "amp D50",

    # Nutrition & diet
    "carbohydrate", "diabetic diet", "glucose control"
]
candidate_sentences = [
    s for s in sentences if any(term.lower() in s.lower() for term in diabetes_terms)
]
print(f"Candidate diabetes-related sentences found: {len(candidate_sentences)}")

# Compute similarity
matches = []
for sent in candidate_sentences:
    emb = model.encode(sent, convert_to_tensor=True)
    similarities = util.cos_sim(emb, ref_embeds)
    score = torch.max(similarities).item()
    if score > 0.5:
        matches.append((sent, round(score, 3)))

# Display results
if matches:
    print(f"\n✅ Found {len(matches)} diabetes-relevant sentences:\n")
    for sent, score in matches:
        print(f"[score={score}] {textwrap.fill(sent, width=100)}")
        print("=" * 120)
else:
    print("⚠️ No diabetes-relevant sentences found (consider lowering the threshold).")