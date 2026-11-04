# Strategy and running record

Initial plan: inspect the teacher before computing on any training photograph. Read its architecture and conv1 weight summaries first. Next interpret its first-layer filters algebraically (colour sums, edge moments), and use teacher interventions on training images to distinguish dependence on colour, texture and geometry. Build only analytic, independently chosen image features and semantic decision rules. Never transplant teacher numbers or fit rules to teacher outputs. Training feature summaries may diagnose properties, but no automated parameter selection or learned tables will be used.

Planned validation: use all 25 permitted passes on documented, manually designed alternatives, relying on train diagnoses between changes. No own scripts read validation images. Target is 68.90%, but it is a target rather than an achieved measurement.

Teacher inspected first: teacher/README.md, layer listing and conv1 weight summaries. Architecture uses local colour/edge filters, repeated spatial pooling and global average evidence. No image pixels have yet been inspected. Next work will examine filter shapes and class-preserving image interventions; this will guide colour opponency, texture and shape features.

No classifier measurements yet. No delegation yet.

Delegated three train-only investigations to collaboration agents organic (tench/dog/saw), geometry (church/cassette/truck/pump), round_sky (horn/golf/parachute). They may propose analytic features in separate files and must inspect teacher, never view images or read validation, fit parameters, or delegate further. Root retains classifier and validation control.

Teacher inspection: inspect_teacher.py examines three sorted training images per class under original/grayscale/Gaussian blur/quarter-turn, including post-ReLU activations. Colour removal destroys all three sampled tench decisions and weakens horns; cassette/truck/pump retain most grayscale decisions. Blur destroys all sampled machine/building/tench decisions but preserves sampled golf and parachute decisions. Quarter-turn damages machine decisions much more than fish or round-object decisions. Conv1 algebra reveals explicit colour-opponent fields and axis derivative filters. Consequence: implement independent hue/opponent masks, spatial pooling, oriented gradients, local texture, and object geometry; do not transfer any values or activation vectors.

v01: initial analytic colour and edge feature baseline in classify.py. diagnose.py reports class feature medians for explanation only (not written as parameter tables). Initial failure diagnosis: broad neutral/axis scores dominate organic classes, almost all dogs and fish routed to cassette. Need gated rather than raw colour evidence and object-level shapes. Official measurements pending below.

v01 official: train 105/505 = 20.79%; validation pass 1: 122/495 = 24.65%, 8.2 ms/image, no errors. First scoring attempt rejected numeric comment coverage, fixed comments before actual scoring; rejection used no pass. Baseline confirms colour/neutral scores need balanced bounded votes rather than unbounded area combinations. No claim of improvement near target.

Change plan v02: semantic unit evidence at explicitly chosen visible fractions, plus independent coat adjacency, olive/smooth scales, coherent saw housing, warm fine brass structure, lower-field sky, and long axis-aligned contours. Each feature is justified by teacher interventions and measured class separation, not by activation fitting. Organic agent found horizontal olive-body shape weak and will not use it. Round agent found Hough circles nonseparating and removed them. Geometry agent found paired tyres/repeated contour windows weak and is removing them. These negative results prevent adding attractive but unsupported detectors.

Restart continuation: prior investigation files are available; no agents survived. Delegated fresh train-only organic_v2, geometry_v2, round_v2 investigations, each instructed to inspect teacher, prohibit fitting and validation, and write separate reviewed features/reports. Root will first implement the already planned bounded-evidence v02 for a usable diagnostic baseline. Existing feature summaries show cassette has long horizontal lines, pump upright lines, church lower detail, horn warm fine detail, and parachute lower-field blue. White circle compactness alone is weak, so it cannot be a dominant ball vote.

v02 bounded material/structure votes: train 185/505 = 36.63%, 7.1 ms/image, zero errors. Per-class successes (class order): 25,23,27,6,22,13,20,16,9,24. Fish/dog now receive decisions, but saws still go fish/dog/player and golf goes fish. Pump is confused with cassette. Teacher organic investigation independently finds generic parallel-bar evidence responds more to churches, so reject bare line-pair detector and require a housing relationship. Validation result below.
val passes: 1 of 25 used; 24 left (this run costs one if it is scored)
program: classify.py  size: 9542 bytes
comment coverage: 53/85 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.3980 (197/495)
errors: 0 (timeouts: 0)  mean ms/img: 8.5 (excl. timeouts)  wall: 0.7s
val pass 2 of 25 used; 23 left


v03 plan: material_probe.py tests red-green proximity as a material boundary: requiring olive red no more than a small visible gap below green disproportionately removes golf grass, retaining more fish material. The threshold .025 was chosen as a small visible unit-RGB channel difference before analysis, not searched. Isotropic Hessian dark pits in white material do NOT separate golf from other classes; reject that detector. Change only the olive mask to the stricter near-equal-red-green material boundary, plus require red not greatly above green to reject warm brown fur.

v03 measured: train 185/505 = 36.63%, 12.6 ms/image, zero errors. Golf rises 9->14 successes but fish falls 25->18; no overall gain. Validation pass 3: 194/495 = 39.19%, 17.6ms, zero errors. Difference from v02 is too small to interpret. Retain strict material definition provisionally because it excludes visible grass, with body-shape evidence still needed.
Teacher_structure.py: one separate training sample/class, 128/32/16/8-pixel tile shuffles, read conv3 activations and top decisions. Fish survives 32px tile shuffling but not finer; dog/cassette fail even 32px shuffling, horn survives 16px, truck survives 8px. Supports object/patch relationships for dog/player and local material/tube texture for horn; a generic colour/edge histogram cannot reproduce all evidence. No activations/constants copied.

Delegation return: geometry_v2.py / geometry_report.md tests all train, teacher vertical flips/quadrant permutation/axis blur, including conv3/conv5 summaries. Four-way machine/building vote alternative old95/205 vs new96/205 (insignificant). Ablation confirms chassis/display affect relevant images but cause cross-class mistakes. Roof, tyre-ring and generic cabinet pairs nonselective. Root integration v04 will use normalized chassis only when rigid structure exists, and upper-screen-over-light-body only as one corroborating pump vote. Root contour_probe.py confirms multilevel bright near-circle components fail: pumps trigger more than golf; no circle detector added.

v04 measured: train185/505=36.63%,14.1ms, zero errors. Pump16->20, cassette27->28, truck20->15; overall unchanged. 
val passes: 3 of 25 used; 22 left (this run costs one if it is scored)
program: classify.py  size: 17425 bytes
comment coverage: 102/145 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.3919 (194/495)
errors: 0 (timeouts: 0)  mean ms/img: 13.0 (excl. timeouts)  wall: 1.0s
val pass 4 of 25 used; 21 left

Round delegation returned round_v2.py/report: teacher warm-pixel selective blur and blue-to-neutral support local warm detail for horns, broad low-frequency lawn/sky for golf/parachute. All-train fixed smooth-lawn+white+low-edge trigger golf10 with five non-golf hits; warm detail horn27 but church19. v05 will replace broad brass with bright warm detailed material, use sparse-edge smooth lawn, and allow surrounding gray/cloud sky with blue corroboration, not bare neutral background. Shared gradient/color work will be reused.

v05 measured train195/505=38.61%,16.2ms, no errors; horn13->19, golf14->19, parachute24->29.
val passes: 4 of 25 used; 21 left (this run costs one if it is scored)
program: classify.py  size: 19751 bytes
comment coverage: 115/159 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4061 (201/495)
errors: 0 (timeouts: 0)  mean ms/img: 15.8 (excl. timeouts)  wall: 1.1s
val pass 5 of 25 used; 20 left

Root texture_probe.py explores standard local structure tensors at 2/5/10-pixel scales and conditional fine/broad contrast. Orientation coherence strongest on audio/pump, lowest fish, but no decisive novel separation; not adding this bank yet. v06 plan: replace generic dog adjacency with adjacent dark/brown material around broad white interiors, morphology radius representing ~7% frame width. Broad white fur survives, narrow brass highlights vanish. Agent's v03-only causal trial 185->191/505, with dog false positives84->60 but true dogs24->22: weak aggregate gain, good mechanism specificity. Agent organic_v2.py also has face template and bar+housings; both weak, not integrated.

v06 measured train197/505=39.01%,22.5ms, no errors.
val passes: 5 of 25 used; 20 left (this run costs one if it is scored)
program: classify.py  size: 20826 bytes
comment coverage: 121/164 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4101 (203/495)
errors: 0 (timeouts: 0)  mean ms/img: 42.0 (excl. timeouts)  wall: 2.8s
val pass 6 of 25 used; 19 left

No material improvement over v05. score_probe.py diagnoses flat votes: saw evidence is much weaker than all other classes because only large coherent orange housing counts, despite photos often including the much longer bar and distant motor. High-confidence scene gates are mostly already correctly classified, so a cascade alone cannot fix the main deficit. Delegated next train-only investigations: saw_v3 (finite guide bar/casing shape), machine_v3 (player/pump panel layout), texture_v3 (photographic/material texture), each no val, no main edits, at most two fixed semantic candidates.
v07 plan: recognise a resolved but distant orange motor: quarter-image width by sixteenth-image height is 1/64 area; two such chromatic pieces total1/32 area. These geometric fractions replace the requirement of a4%-area solid orange patch, while retaining vibrant orange material, dark structure and rigidity subtraction. No parameter search, only this one physical-size hypothesis.

v07 measured train196/505=38.81%,26.0ms, no errors; saw6->13 but several other classes drop.
val passes: 6 of 25 used; 19 left (this run costs one if it is scored)
program: classify.py  size: 20897 bytes
comment coverage: 121/164 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4061 (201/495)
errors: 0 (timeouts: 0)  mean ms/img: 20.1 (excl. timeouts)  wall: 1.4s
val pass 7 of 25 used; 18 left

Second delegation round: texture_v3 alternatives both194/505 versus v06 197/505, rejected. Ridge bonus horn+4 correct but7 lost elsewhere; lawn bonus golf+3 but6 lost, four of them trucks. Teacher selective lawn blur preserves truck while global blur changes truck to golf; coherent manufactured structure is essential counterevidence. Full details texture_v3_report.md. Machine component/control trial197/505 (same v06 total), adding portrait/landscape vote202/505. Those global votes harm churches/trucks despite recovering players/pumps. Root v08 will gate aspect-preserving component/control adjustments to the existing player-vs-pump contest: object-shape corroboration should not override a truck/church decision. Exclude image-orientation vote for now because object geometry is better grounded. Start v08 from v06 (current best overall), discarding v07's broad small-orange change.
Root fish_probe.py: finite olive ellipses alone nonselective, pink-plus-body contrast somewhat higher in fish but overlapping dogs/saws. No rule added; fish_teacher.py examines pink neutralisation for causal support.

v08 measured train200/505=39.60%,38.2ms, no errors; player27->25, pump16->21, other classes unchanged.
val passes: 7 of 25 used; 18 left (this run costs one if it is scored)
program: classify.py  size: 28783 bytes
comment coverage: 167/222 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4141 (205/495)
errors: 0 (timeouts: 0)  mean ms/img: 36.5 (excl. timeouts)  wall: 2.4s
val pass 8 of 25 used; 17 left

v09 plan from causal evidence: fish_teacher.py neutralises the independently defined pink-skin mask while preserving brightness/edges. On three new fish, target margin decreases each time; on three dogs, it stays or rises; on three saws changes are much smaller. Fish_probe.py also shows finite-olive-body plus nearby pink stronger for fish than isolated global skin, albeit overlapping other classes. Replace broad warm skin by actual pink RGB ordering (blue retained, unlike brass), and add ONE corroborating finite-body/near-skin vote via analytic ellipse-minus-surround filters at three meaningful body scales and level/30-degree tilts. No parameter search; a tenth skin occupancy times half-body colour support defines full relation evidence. This is weak conjunction corroboration, not a hard fish detector.

v09 measured train204/505=40.40%,118.9ms, no errors; fish19->26 but collateral losses.
val passes: 8 of 25 used; 17 left (this run costs one if it is scored)
program: classify.py  size: 31633 bytes
comment coverage: 183/241 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4101 (203/495)
errors: 0 (timeouts: 0)  mean ms/img: 166.3 (excl. timeouts)  wall: 10.6s
val pass 9 of 25 used; 16 left

Validation unchanged within noise and mean166.3ms exceeds our performance aim. Discard expensive held-body ellipse bank; retain it in v09 for provenance. Negative ellipse result plus several weak score iterations is a local-minimum signal. v10 is a genuinely different combination strategy: fuzzy conjunctions of essential material/structure cues, followed by within-machine cabinet/control resolution. All classes use a unit-range essential evidence score rather than arbitrary sums of correlated colours. E.g fish requires olive with skin/green support, horn requires warm fine detail, player requires rigidity plus broad horizontal structure, golf requires lawn/white. Constants inherited by physical meanings, no automated fitting/search. Specific pink-skin mask retained from causal probe, without body filters.

v10 measured train182/505=36.04%,44.7ms, no errors. Conjunction pipeline improved saw6->15,dog22->25,golf19->22,para28->30 but truck16->3,pump21->6,fish19->13.
val passes: 9 of 25 used; 16 left (this run costs one if it is scored)
program: classify.py  size: 29139 bytes
comment coverage: 170/226 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.3556 (176/495)
errors: 0 (timeouts: 0)  mean ms/img: 82.7 (excl. timeouts)  wall: 5.4s
val pass 10 of 25 used; 15 left

Reject stand-alone v10: missing conjunction factors disproportionately erase correct machine evidence. v11 plan is a two-stage semantic hierarchy: existing broad structure votes retain church/player/truck/pump choices; when these do not win, use v10's unit-range material conjunctions among the remaining six organic/aerial/tool classes. This tests whether local-material interactions help without requiring every machine part to be visible. No coefficients or thresholds selected by search; both formulas are fully specified existing hand rules.
Saw delegation v3 return saw_report.md: finite guide-bar, closed handle, uniform-border silhouette and hand-drawn brand lettering all failed coverage/separation; none integrated. Sky broad-canopy route200/505 unchanged v08, rejected. A canopy plus smaller payload BELOW relation is selective (11para,6other fixed triggers) and will be considered separately once report arrives.

v11 measured train198/505=39.21%,36.3ms,no errors. Hierarchy preserves machine sensitivities but fish loses6, horn loses3 while saw+6: no net improvement.
val passes: 10 of 25 used; 15 left (this run costs one if it is scored)
program: classify.py  size: 34927 bytes
comment coverage: 203/264 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.3980 (197/495)
errors: 0 (timeouts: 0)  mean ms/img: 38.2 (excl. timeouts)  wall: 2.5s
val pass 11 of 25 used; 14 left

Reject hierarchy and return to v08. v12 adds a qualitatively specific geometric relation: a broad/near-round canopy and much smaller suspended component below it against a smooth sky-coloured per-image border background. Every test is relative geometry/current-image statistics; no stored examples. Teacher foreground/background interventions on six missed paras show both object and sky can matter (foreground retains5, backdrop4, blur4). Agent's fixed relation triggers11para+5other, including4previouslymissed paras. Four semantic votes correspond to background, canopy, small payload, suspension relation; evaluate causal effect before retaining. Share existing thumbnail/gray/gradient arrays.

v12 measured train203/505=40.20%,35.8ms,no errors; para28->32,dog22->21,others unchanged.
val passes: 11 of 25 used; 14 left (this run costs one if it is scored)
program: classify.py  size: 32786 bytes
comment coverage: 192/255 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4101 (203/495)
errors: 0 (timeouts: 0)  mean ms/img: 38.1 (excl. timeouts)  wall: 2.5s
val pass 12 of 25 used; 13 left

No significant val gain, but retain the specific payload relation provisionally as it changes the intended four missed paras and costs one correct dog. v13 plan: weak photographic-composition vote only within player/pump contest. Players' broad cabinets usually invite landscape framing; tall dispensers invite portrait framing. One extra vote is smaller than a matched two-vote object component. The frame aspect itself is measured anew, no filename/metadata lookup, no inferred sample-specific rule. Existing teacher tall/wide stretch intervention supports respecting shape, while this framing vote remains context evidence.

v13 measured train208/505=41.19%,34.8ms,no errors; cassette25->28,pump21->23.
val passes: 12 of 25 used; 13 left (this run costs one if it is scored)
program: classify.py  size: 33430 bytes
comment coverage: 196/259 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4101 (203/495)
errors: 0 (timeouts: 0)  mean ms/img: 42.3 (excl. timeouts)  wall: 2.8s
val pass 13 of 25 used; 12 left

Val unchanged despite intended machine-pair training effect. v14 outdoor structural context: outdoor_probe.py verifies smooth BLUE in the upper half (not neutral studio background, not arbitrary paint) separates many trucks/pumps mistaken for indoor players: chosen tenth-upper-field trigger catches9false-audio trucks and5false-audio pumps, no currently correct audio. Add one weak outdoor countervote to cassette, and let the truck's existing outdoor-context vote use either green or smooth upper blue. Quarter-upper-field occupancy saturates clearly visible sky. This changes context only, still requiring machine body/chassis evidence. Previous teacher grayscale and spatial interventions show structure remains essential, so sky must not classify trucks by itself.

v14 measured train211/505=41.78%,38.1ms,no errors; trucks16->25,with small losses in audio/pump/church/para.
val passes: 13 of 25 used; 12 left (this run costs one if it is scored)
program: classify.py  size: 33660 bytes
comment coverage: 197/260 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4283 (212/495)
errors: 0 (timeouts: 0)  mean ms/img: 65.4 (excl. timeouts)  wall: 4.3s
val pass 14 of 25 used; 11 left

Currently best validation42.83%, still far below target and within small-step noise. Church skyline delegation returned no corrections in fixed200/505 trial, rejected (church_v4_report.md). Root held_material_probe.py tests contextual mask growth near pink hands; it recovers olive in fish but equally in saw/horn, so reject growth. v15 instead addresses causal context logically: fish material must coexist with green surroundings OR pink skin. Broad olive with neither is often fur or machinery, and context should not merely add independent votes to a missing body. Gate the three material/smoothness votes by max(green,pink) support, using fixed .15 vegetation area and .10 skin area meanings, and replace generic warm-skin vote with the pink-specific mask proven by teacher intervention. No shape-bank runtime cost.

v15 measured train214/505=42.38%,36.2ms,no errors; fewer false fish (dog8->3,saw13->8,golf8->4),fish19->18,net+3.
val passes: 14 of 25 used; 11 left (this run costs one if it is scored)
program: classify.py  size: 34236 bytes
comment coverage: 200/263 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4283 (212/495)
errors: 0 (timeouts: 0)  mean ms/img: 42.5 (excl. timeouts)  wall: 2.8s
val pass 15 of 25 used; 10 left

Val unchanged. v16 tests a reflective-metal representation rather than merely adding texture votes: replace TWO of the horn's three bright-fine votes by analytic warm ridge/trough votes. The previous agent +1-ridge trial failed by overcalling warm materials, so total positive horn vote budget stays five. Tube curvature uses per-image Hessian eigenvalues: strong across-tube curvature and weak along-tube curvature (3:1), in warm material that can be darker than bright gold. Conditional half-material ridge support and tenth-frame total ridge support define resolved coiled tubes. Prior teacher warm-only blur causally weakens both ridge response and horn evidence on two recovered horns. No filters/values transferred from teacher.

v16 measured train213/505=42.18%,38.5ms,no errors (one fewer horn thanv15).
val passes: 15 of 25 used; 10 left (this run costs one if it is scored)
program: classify.py  size: 35958 bytes
comment coverage: 211/274 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4303 (213/495)
errors: 0 (timeouts: 0)  mean ms/img: 41.4 (excl. timeouts)  wall: 2.7s
val pass 16 of 25 used; 9 left

43.03% best val by one image, not a demonstrated improvement. v17 changes the overall view geometry rather than adding votes: central square crop at the original shortest-side length, then160 resize, replacing full-image square warping for shared material/edge/sky features. This preserves physical circle/body/cabinet proportions and puts the subject at a consistent scale. Original aspect-preserved machine stage and coarse full-scene geometry remain separate shared context stages. Teacher's crop pipeline and tall/wide interventions motivated respecting object proportions; no teacher preprocessing numeric constants are copied. Compare all-train sensitivity and then val.

v17 measured train221/505=43.76%,37.0ms,no errors; crop gains dog+2,audio+2,horn+2,truck+6,pump+2,loses church2,golf3,saw1.
val passes: 16 of 25 used; 9 left (this run costs one if it is scored)
program: classify.py  size: 36321 bytes
comment coverage: 213/277 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4465 (221/495)
errors: 0 (timeouts: 0)  mean ms/img: 39.9 (excl. timeouts)  wall: 2.6s
val pass 17 of 25 used; 8 left

Best44.65%, still below target. v18 intentional two-view architecture: preserve central-square object scale/shape while measuring full-frame sky/lawn/facade context once for all classes. The object crop can remove the ground/sky that distinguishes churches and golf, so do not use it for every context feature. Existing full-frame coarse geometry already supplies a lower-detail contrast; use it for facade. The shared full-frame thumbnail/gradient stage provides outdoor colours, smooth lawn, surrounding sky and canopy/payload relation. This is geometric design, not score fitting.

v18 measured train226/505=44.75%,42.6ms,no errors; contextual scene improves some train counts.
val passes: 17 of 25 used; 8 left (this run costs one if it is scored)
program: classify.py  size: 39946 bytes
comment coverage: 236/301 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4384 (217/495)
errors: 0 (timeouts: 0)  mean ms/img: 46.1 (excl. timeouts)  wall: 3.0s
val pass 18 of 25 used; 7 left

Val43.84% does not improve v17. v19 uses equal-vote multi-view consistency rather than assigning each feature to a view: evaluate the frozen v17 hand rules on central aspect-preserving view and full-frame view, average the two semantic score sets per image. Equal half weights express no fitted preference; this is per-image multi-scale inference, not averaging training data. All formulas remain handwritten. Restore v17 feature stage; remove duplicated context implementation. One OpenCV worker prevents tiny-thumbnail threading overhead; do not change predictions.

v19 measured train221/505=43.76%,47.7ms,no errors; equal multi-view changes sensitivities (truck31->26,pump23->26,para31->33) without total train gain.
val passes: 18 of 25 used; 7 left (this run costs one if it is scored)
program: classify.py  size: 37005 bytes
comment coverage: 218/282 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4444 (220/495)
errors: 0 (timeouts: 0)  mean ms/img: 49.1 (excl. timeouts)  wall: 3.2s
val pass 19 of 25 used; 6 left

44.44% is effectively tied with single-viewv17 44.65%; simpler/faster v17 remains best baseline. v20 returnsv17 and adds one weak contour-coat corroboration: a resolved broad white patch with warm fur immediately outside and nonrectangular contour. The feature was separately measured (organic_report.md) with stronger dog response but some horn overlap, so cap at one vote and leave rigid-structure subtraction. A .05 conjunctive support means, for example, half-size resolved patch times one-fifth warm ring times half curved boundary. This is independently described shape/material evidence, not a template or learned value.

v20 measured train214/505=42.38%,24.9ms,no errors; dog24->28 but horn20->14,fish18->16,golf18->16.
val passes: 19 of 25 used; 6 left (this run costs one if it is scored)
program: classify.py  size: 38931 bytes
comment coverage: 229/297 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4404 (218/495)
errors: 0 (timeouts: 0)  mean ms/img: 24.5 (excl. timeouts)  wall: 1.6s
val pass 20 of 25 used; 5 left

Reject coat-shape addition: reflective horn patches also have curved warm borders. v21 tests the constrained guide-bar relation as corroboration, revisiting the largest uncovered class (saw). The old original-orange bar has moderate separation (13saw/34others at fixed diagnostic threshold) but was not tried in the whole classifier. Permit yellow as well as orange motor material, since saw casings may be either, and add at most two relation votes to frozenv17. Require sustained neutral bright bar, paired dark chain borders, substantial parallel overlap, and warm casing localised at an end. The bar relation supplies complementary structure rather than another global colour vote. All geometry remains manual; do not use the failed frontal dog template.

v21 measured train217/505=42.97%,31.7ms,no errors; saw6->17 butchurch22->17,truck31->28,player28->26 and small losses elsewhere.
val passes: 20 of 25 used; 5 left (this run costs one if it is scored)
program: classify.py  size: 41776 bytes
comment coverage: 246/316 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4424 (219/495)
errors: 0 (timeouts: 0)  mean ms/img: 33.2 (excl. timeouts)  wall: 2.2s
val pass 21 of 25 used; 4 left

v22 diagnosis: matched parallel boundaries in a lower architectural facade are not sufficient guide-bar evidence. Gate the bar bonus by absence of the lower facade transition rather than globally increasing saw confidence. Teacher bar_facade_teacher.py checks three newly recovered saws and three false-bar churches (train only): blurring only lower half destroys2/3teacher saw decisions and2/3church decisions while conv3 averages remain similar, showing that meaningful structure is local but object-specific. This does not turn bare lines into saws; sky-to-facade context remains valid contrary evidence. Use existing unit facade as a soft veto, no new threshold.

v22 measured train222/505=43.96%,62.8ms,no errors; architecture veto restores all five lost churches but saw17->15.
val passes: 21 of 25 used; 4 left (this run costs one if it is scored)
program: classify.py  size: 41806 bytes
comment coverage: 246/316 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4424 (219/495)
errors: 0 (timeouts: 0)  mean ms/img: 88.3 (excl. timeouts)  wall: 5.9s
val pass 22 of 25 used; 3 left

Val44.24%, tied withv21 and below simplerv17; reject guide-bar branch for final due cost and no demonstrated gain. v23 returns v17 and revisits shallow-depth-of-field golf context ONLY with explicit structural/sky vetoes. Prior agent ungated lawn bonus recovered3golf but broke4trucks; teacher green-only blur preserves truck while full blur destroys truck, so background blur must not override rigid structure. Add up to two green-material fine-residual votes, multiplied by absent rigidity, absent facade and absent blue sky. This preserves object constraints and uses shared residual rather than an extra filter bank. All scales from the earlier semantically specified candidate, no search. Delegated finalv17 code-only compliance/comment/dead-code audit to geometry_v2, no measurements or image access.

v23 measured train220/505=43.56%,25.1ms,no errors; only1golf recovered while2trucks lost despite structural gate.
val passes: 22 of 25 used; 3 left (this run costs one if it is scored)
program: classify.py  size: 37254 bytes
comment coverage: 218/283 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4444 (220/495)
errors: 0 (timeouts: 0)  mean ms/img: 47.9 (excl. timeouts)  wall: 3.1s
val pass 23 of 25 used; 2 left

Reject extra lawn texture; v17 still best/simple. v24 last new hypothesis: distant orange housing fromv07 should be allowed only without competing architectural/coiled-metal interpretation. Reuse the independently chosen1/64 single patch and1/32 total patch scales (quarter-width by sixteenth-height casing), but apply only the EXTRA small-housing evidence beyond existing large-housing votes, multiplied by absence of facade and absence of warm tube geometry. Large clear housings retain original evidence. This combines structural counterevidence rather than sweeping size thresholds; no search and no teacher fitting. Final pass25 is reserved for chosen audited deliverable.

v24 measured train216/505=42.77%,25.6ms,no errors; small housing recovers2saws with collateral losses5 net versusv17.
val passes: 23 of 25 used; 2 left (this run costs one if it is scored)
program: classify.py  size: 37009 bytes
comment coverage: 217/281 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4485 (222/495)
errors: 0 (timeouts: 0)  mean ms/img: 27.7 (excl. timeouts)  wall: 1.9s
val pass 24 of 25 used; 1 left

Highest measured val44.85% by one image overv17, far below meaningful separation. Provisional final selectionv24 uses extra small-case semantics but does not claim demonstrated improvement. v25 will verify a code-cleaned equivalent, not tune further. Target68.90% not achieved. Final code-only audit is in progress; no training/validation images are involved in audit.

## Consolidated official measurements

Each vNN snapshot is classify_vNN.py; full training confusion/predictions are train_vNN.json/.csv and val only has official aggregate logs. No own script has read any validation image. Numeric/comment rejections above consumed no passes. Agent-only feature and ablation measurements are in the named reports referenced in this notebook.

|Version|Train correct/505|Train accuracy|Val correct/495|Val accuracy|Val ms/image|
|---|---:|---:|---:|---:|---:|
|v01|105|20.79%|122|24.65%|8.2|
|v02|185|36.63%|197|39.80%|8.5|
|v03|185|36.63%|194|39.19%|17.6|
|v04|185|36.63%|194|39.19%|13.0|
|v05|195|38.61%|201|40.61%|15.8|
|v06|197|39.01%|203|41.01%|42.0|
|v07|196|38.81%|201|40.61%|20.1|
|v08|200|39.60%|205|41.41%|36.5|
|v09|204|40.40%|203|41.01%|166.3|
|v10|182|36.04%|176|35.56%|82.7|
|v11|198|39.21%|197|39.80%|38.2|
|v12|203|40.20%|203|41.01%|38.1|
|v13|208|41.19%|203|41.01%|42.3|
|v14|211|41.78%|212|42.83%|65.4|
|v15|214|42.38%|212|42.83%|42.5|
|v16|213|42.18%|213|43.03%|41.4|
|v17|221|43.76%|221|44.65%|39.9|
|v18|226|44.75%|217|43.84%|46.1|
|v19|221|43.76%|220|44.44%|49.1|
|v20|214|42.38%|218|44.04%|24.5|
|v21|217|42.97%|219|44.24%|33.2|
|v22|222|43.96%|219|44.24%|88.3|
|v23|220|43.56%|220|44.44%|47.9|
|v24|216|42.77%|222|44.85%|27.7|

Final audit return: classify_audited.py and audit_report.md. Agent performed code-only static equivalence audit ofv17, removed provably unused feature outputs/calculations, expanded/corrected comments, addedcv2.setNumThreads(1). No images read/no measurements/no teacher calls. Root copied the exact manually writtenv24 small-case bonus into the cleanedv17 source, giving chosen finalv25. Corrected two explanatory comments (two rigid penalty votes, fifteen-percent metal area). Before consuming final validation pass, verify every training prediction matches frozenv24; no parameter or behavior changes intended.

Finalv25 equivalence check: all505training predictions and errors match frozenv24 exactly. Static rule/comment checks pass;223comment lines/258code lines, zero uncovered numeric lines. Train216/505=42.77%,29.5ms/image, no errors/timeouts.
val passes: 24 of 25 used; 1 left (this run costs one if it is scored)
program: classify.py  size: 36496 bytes
comment coverage: 223/258 lines; 0 numeric lines without a nearby comment
images: 495  accuracy: 0.4485 (222/495)
errors: 0 (timeouts: 0)  mean ms/img: 23.5 (excl. timeouts)  wall: 1.6s
val pass 25 of 25 used; 0 left

Final deliverable classify.py (also classify_v25.py): central aspect-preserving material view, full-scene coarse structure, aspect-preserving machine components/control rows, analytic colour/texture/curvature masks, geometric canopy-payload relation, bounded semantic votes, conservative machine-pair refinement and casing-scale counterevidence. OnlyPILreads the supplied image. No learned model, teacher import, stored statistics, exemplars, data-derived arrays, network, environment or extra file access.

Final result: train42.77%, validation44.85%; all25validation passes used. Target68.90% NOT achieved. Validation differences among late candidates are below sampling uncertainty; one-image advantage is not a demonstrated hidden-test gain. No hidden-test score is claimed. Largest unresolved deficit is saw recognition and reliable fish/golf/horn separation. Many plausible geometric/texture detectors failed class separation and were rejected, with reports retained. Final code comments explain each material, geometry operation, scale and vote.

Final table row: v25 train 216/505 (42.77%), val 222/495 (44.85%), 23.5ms/image, zero errors/timeouts.
