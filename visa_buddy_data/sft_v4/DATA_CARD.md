# SFT data card (Visa Buddy v4: Qwen3-4B, extract + answer)

- train rows: **862** {'answer:situation': 60, 'answer:answer': 313, 'extract:answer': 313, 'extract:no_quote': 64, 'extract:policy': 52, 'extract:situation': 60}
- valid rows: **99** {'extract:answer': 31, 'extract:situation': 8, 'answer:situation': 8, 'answer:answer': 31, 'extract:no_quote': 14, 'extract:policy': 7}
- dropped: {'answer:answer_not_grounded': 199, 'no_quote:teacher_found_quotes': 463, 'answer:teacher_quotes_only': 43, 'answer:teacher_refused_no_quotes': 65, 'answer:teacher_refused_policy': 1, 'situation:teacher_refused_no_quotes': 9, 'situation:answer_not_grounded': 47, 'situation:teacher_quotes_only': 6, 'policy:teacher_refused_no_quotes': 62, 'policy:teacher_answered': 19}

## Checks

- [x] no_eval_question_in_any_row
- [x] no_eval_page_as_source
- [x] valid_and_train_share_no_page
- [x] extract_targets_are_valid_json
- [x] refusal_share_of_extract_ge_0_20

## Teacher and gate

Teacher: Qwen3-30B-A3B-Instruct-2507 running the v4 pipeline. It missed the pre-set teacher gate by one
refusal (30/32 vs >= 31) and beat the untuned student on fact coverage, over-refusal and wrong rules;
the data is used under a documented exception (reports/V4_PLAN.md, "Changes"). Refusal rows are kept
only where the teacher refused; 19 policy questions it answered were dropped.

## Manual review (before training)

20 random train rows were read (a different sample from the one below). Every extract target is
verbatim canada.ca text (checked by the pipeline), and no answer adds a fee, date or rule missing from
its quotes; Yes/No openings match their quotes. Situation rows pick the case asked about ("fail the test
all 3 times -> hearing"). Noise found: one extract target is page navigation text (link titles), and
one quote starts with the page's own "Answer:" label. Both are real page text; left in.

## 20 random train rows

### extract / answer (work_permit)
**Q:** What should I write in the Job Title box for Question 4 when filling out the IMM 5710 form to extend my open work permit as a Quebec investor?

**Target:** {"quotes": [{"source": 1, "text": "Answer: Write “T10 Quebec Investors” in the Job Title box."}]}

### extract / no_quote (work_permit)
**Q:** Do I need to pay a fee for the open work permit holder when applying for a work permit as a temporary resident permit holder?

**Target:** {"quotes": []}

### answer / answer (work_permit)
**Q:** Can I get a work permit as a self-employed fishing guide on a Canadian lake, and do I need a Labour Market Impact Assessment?

**Target:** Yes, you can get a work permit as a self-employed fishing guide working on a Canadian lake. You may be exempt from needing a Labour Market Impact Assessment if your work will either attract tourism or benefit Canadian citizens or permanent residents.

### answer / answer (citizenship)
**Q:** Can I take the citizenship test in person or with Microsoft Teams if I have a weak Internet connection?

**Target:** Yes, we may invite you to take the test in person or with Microsoft Teams if you have a weak or unreliable Internet connection.

### answer / answer (family)
**Q:** I have a PR application in process and am experiencing family violence in Canada. How can I inform IRCC about my situation?

**Target:** You can inform IRCC about your situation by submitting a web form. In the form, let us know that you have a PR application in process, are currently living in Canada, and are requesting consideration due to family violence. You should also upload a letter with details of your situation and any additional supporting documents.

### extract / answer (express_entry)
**Q:** Can I get a refund of the processing fee if my Express Entry application is incomplete?

**Target:** {"quotes": [{"source": 3, "text": "If your application isn’t complete, we’ll tell you and refund your:\nprocessing fee\nright of permanent residence fee (if you’ve paid it)"}]}

### answer / situation (work_permit)
**Q:** I'm an unpaid intern coming to Canada through the International Co-op program. I've been told I need to show proof of funds. Do I need to show enough money for just the first 3 months or for my entire stay in Canada?

**Target:** Yes, if you’re an unpaid International Co-op (Internship) participant, you may need to show proof of additional financial resources to cover expenses for your entire stay.

### extract / no_quote (citizenship)
**Q:** What kind of identification do I need to take the citizenship test?

**Target:** {"quotes": []}

### extract / no_quote (family)
**Q:** Can I still apply under the public policy for undeclared family members after September 10, 2026?

**Target:** {"quotes": []}

### extract / answer (work_permit)
**Q:** Can I participate in the International Experience Canada Working Holiday program more than once if I'm an Australian citizen and I participated in IEC before 2015?

**Target:** {"quotes": [{"source": 5, "text": "Exception: If you participated in IEC one or more times before 2015, you might be eligible for a final participation of up to 24 months in the category of your choice."}]}

### extract / policy (out_of_scope)
**Q:** I'm preparing for an immigration court case. What legal strategies would you recommend?

**Target:** {"refuse": true}

### extract / answer (study_permit)
**Q:** How can I find work and plan my career in Canada as a graduate student?

**Target:** {"quotes": [{"source": 5, "text": "Find work and plan your career with Canada’s national employment service"}, {"source": 5, "text": "Virtual sessions: Pathways to permanent residence"}, {"source": 5, "text": "Register for a session to learn about pathways to stay in Canada after you graduate"}]}

### extract / situation (citizenship)
**Q:** I failed the citizenship test twice already. If I fail again, what happens next? I want to make sure I understand the process before my next attempt.

**Target:** {"quotes": [{"source": 4, "text": "If you fail the test all 3 times, we’ll invite you to a hearing with a citizenship official."}]}

### answer / answer (study_permit)
**Q:** How long do I have to give my biometrics after receiving the biometric instruction letter to get faster processing for my doctoral degree student application?

**Target:** You must give your biometrics within 2 weeks of getting your BIL to get faster processing.

### extract / policy (out_of_scope)
**Q:** Can you explain the different streams under the BC PNP and which one might be suitable for me?

**Target:** {"refuse": true}

### answer / answer (work_permit)
**Q:** What details should be included in the letter from my employer to qualify for faster processing of my work permit for a medical residency program?

**Target:** Your letter should include the position you’re being offered, including your specialty; the length of the residency or fellowship with beginning and end dates; the location where you’ll be doing the residency or fellowship; your income and a statement that it is comparable to what a Canadian would be paid for the same duties; a statement that your income is not funded by the Canadian public; whether the position is under a collective agreement; and whether you need to be licensed by the provincial College of Physicians and Surgeons. The letter must be on university letterhead and signed by a s

### extract / answer (work_permit)
**Q:** Can I participate in the International Experience Canada Working Holiday program twice if I'm a Dutch citizen?

**Target:** {"quotes": [{"source": 5, "text": "Note: As a Dutch citizen, you may participate in IEC twice. You must choose a different category each time.\nTo be eligible for the Working Holiday category, you must:"}]}

### extract / answer (family)
**Q:** Can I withdraw my sponsorship application for the relative sponsorship program before my relative becomes a permanent resident?

**Target:** {"quotes": [{"source": 4, "text": "You can withdraw your sponsorship application any time before the relative you’re sponsoring becomes a permanent resident of Canada."}, {"source": 2, "text": "You can request to withdraw your sponsorship application any time before the people you’re sponsoring become permanent residents."}, {"source": 3, "text": "You can request to withdraw your sponsorship application any time before the person you’re sponsoring becomes a permanent resident of Canada."}]}

### extract / answer (pgwp)
**Q:** Can I still be eligible for a PGWP if I can't complete all sections of my language test due to a disability?

**Target:** {"quotes": [{"source": 4, "text": "If you can’t have all 4 language abilities assessed due to a disability, you can still be eligible for a PGWP."}]}

### answer / answer (family)
**Q:** Can I withdraw my sponsorship application for the relative sponsorship program before my relative becomes a permanent resident?

**Target:** Yes. You can withdraw your sponsorship application any time before the relative you’re sponsoring becomes a permanent resident of Canada.

