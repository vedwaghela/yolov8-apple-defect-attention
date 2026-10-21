# COMPSCI 764 Recording Guide

Target: 9 minutes of speech. Manish: slides 1-5, 4:30. Ved: slides 6-10, 4:30. Slide 11 contains linked references. Timings are rehearsal targets, not guaranteed narration durations.

The PowerPoint contains the same scripts in speaker notes. Revise the wording into your own voice and verify that you can explain every claim.

## Slide 1: Attention mechanisms for apple defect detection

Presenter: Manish. Target: 25 seconds.

Hello, we are Manish Gadia and Ved Waghela. Our project compares attention mechanisms for apple surface defect detection. We ask whether adding attention to the same YOLOv8n detector improves localisation enough to justify its computational cost. I will explain the problem and our approach, and Ved will present the experiments and findings.

Delivery cue: Introduce both names. Emphasise that this is a comparison of existing attention mechanisms.

## Slide 2: Subtle defects can resemble normal peel texture

Presenter: Manish. Target: 60 seconds.

Consider this apple. A large damaged patch may be easy to notice, but a weak scratch can blend into normal peel texture. Colour variation and reflections also make it difficult to decide exactly where a defect starts and ends. This matters because an automated inspection system needs useful evidence about damaged regions, rather than only a label for the entire fruit. Our task therefore predicts bounding boxes and defect classes. It differs from classifying an apple as good or bad. In this project, we focus on scratches and pest damage in the SSDA dataset. Reliable localisation could support later inspection or grading decisions, although we do not evaluate a complete commercial grading system. The challenge we investigate is whether feature reweighting helps an existing detector distinguish these subtle regions.

Delivery cue: Point at a subtle region in the image. Explain localisation before introducing model terminology.

## Slide 3: Existing detectors and attention mechanisms

Presenter: Manish. Target: 65 seconds.

Existing work already addresses apple-defect detection. HAFREE is a detector designed around subtle apple defects, and its associated dataset gives us a relevant test setting. For our own comparison, we use the compact YOLOv8n implementation. Attention is also an established idea, so our contribution is not to invent a new attention module. Squeeze-and-Excitation, or SE, summarises each feature channel and learns how strongly to weight it. CBAM adds spatial weighting after channel weighting, allowing different image locations to receive different emphasis. Coordinate Attention preserves directional information through separate height and width summaries. These mechanisms provide different ways to alter the feature representation. Published improvements in other settings do not tell us which is best in our configuration, because datasets, training budgets and insertion points can differ. That motivates a matched comparison.

Delivery cue: Define a feature channel as a learned feature map. Avoid claiming these modules have never been compared before.

## Slide 4: Research question and working hypothesis

Presenter: Manish. Target: 65 seconds.

Our question is which attention module offers the best trade-off between detection performance and computational cost when added to this YOLOv8n backbone. The working hypothesis is that selective feature weighting can help the detector separate weak damage from surrounding texture. Because a defect occupies a particular location, we also expect spatially informed mechanisms to be plausible candidates. However, more attention operations can increase execution time, and improved features do not guarantee improved detection. We therefore compare a baseline, SE, CBAM and Coordinate Attention under matched settings. This is an experimental application of existing approaches, rather than a new architecture family. We deliberately narrow the project to detection and measured forward cost. We do not claim robustness to changes in lighting or blur, because those experiments are outside our completed evaluation.

Delivery cue: Stress the connection between local defects and spatial information. Present the hypothesis as something tested, not a proven explanation.

## Slide 5: One attention insertion point in the backbone

Presenter: Manish. Target: 55 seconds.

We insert one attention module after the output of C2f layer six in the backbone. With a 640-pixel input, this stage contains 128 channels at a spatial resolution of 40 by 40, corresponding to stride sixteen. The modified output feeds both the following backbone stages and the existing connection to the neck. The detection head and task remain unchanged. We use the same COCO-pretrained starting weights for the shared architecture and check that the custom modules survive checkpoint saving and loading. This placement retains spatial detail while keeping the comparison manageable. We did not search all possible placements, so our findings apply to this particular configuration. Ved will now explain how we evaluated the four models and what we found.

Delivery cue: Trace the attention output into the following backbone and the existing neck connection, then give the handover and pause. Handover target: 4 minutes 30 seconds.

## Slide 6: A matched evaluation across 12 training runs

Presenter: Ved. Target: 55 seconds.

Thank you, Manish. We used 710 SSDA images containing 1,512 boxes. Our grouped partition contains 500 training, 106 validation and 104 test images. We group related images using similarity and capture-time information to reduce obvious overlap between partitions. This is a precaution, not a guarantee that every physical apple is independent. We trained four models with seeds forty-two, forty-three and forty-four, giving twelve runs of fifty epochs each. All runs share the same input size, optimiser and augmentation policy. Validation performance selects each saved checkpoint, and the held-out test split provides the final comparison. Our primary metric is mean average precision across IoU thresholds from point five to point nine five, which rewards detection and increasingly precise localisation.

Delivery cue: Define IoU as overlap between predicted and annotated boxes. Do not call mAP image-classification accuracy.

## Slide 7: CBAM has the highest mean test mAP50-95

Presenter: Ved. Target: 65 seconds.

This chart shows mean test mAP across the three seeds. CBAM reaches 42.11 percent, compared with 40.70 for the baseline, 41.58 for SE and 41.52 for Coordinate Attention. The CBAM improvement is 1.41 percentage points. Importantly, its paired improvement over baseline is positive for all three seeds, rather than depending on one particularly successful run. The small dots show the individual seeds, the diamonds show the means, and the vertical lines show sample standard deviations. CBAM has the smallest observed variation here, while SE has the largest, but three runs are too few to establish population stability or statistical significance. The ranking also depends on the metric. Coordinate Attention has the highest mean AP at IoU point five, by a small margin. Our main conclusion therefore concerns the stricter primary metric, rather than every possible definition of detection performance.

Delivery cue: Pause on 42.11 and 40.70. Say percentage points, not percent improvement. Point to the individual seeds and explain that the SD lines are not confidence intervals.

## Slide 8: The accuracy gain carries a measurable cost

Presenter: Ved. Target: 55 seconds.

The trade-off becomes clearer when we plot performance against median forward latency. CBAM has the highest primary detection score, but its forward pass takes about 7.113 milliseconds, compared with 6.586 for baseline. That is approximately eight percent more time. SE has a smaller measured overhead of about 1.3 percent, together with a smaller average detection gain. We measured fused FP32 models at batch size one on an RTX 2050, with warm-up and three timing blocks. These timings exclude preprocessing, data transfer and non-maximum suppression, so they are not end-to-end application throughput. CBAM is therefore an accuracy-oriented candidate for this setting. A deployment with stricter latency requirements could reasonably choose another model.

Delivery cue: Explain that upper-left is desirable. Do not translate forward latency directly into real-world grading speed.

## Slide 9: Scratches remain difficult despite the mean gain

Presenter: Ved. Target: 50 seconds.

A higher average score does not mean that every defect is detected. On this fixed test image, the left panel shows annotations and the right shows CBAM predictions. Some upper scratches are missing, while lower damage has overlapping predictions. There is also a prediction near the right edge without a corresponding annotation, which we cannot resolve biologically from the model output alone. Across the test set, CBAM achieves 32.94 percent AP for scratches and 51.28 for pest damage. Scratches remain the harder class. This example illustrates a failure pattern, rather than estimating how frequently it occurs. We also did not inspect attention maps, so we cannot claim that the model demonstrably learned to focus on damaged peel.

Delivery cue: Point to upper misses, then lower overlapping boxes. Avoid changing thresholds while explaining the example.

## Slide 10: A conditional benefit from CBAM attention

Presenter: Ved. Target: 45 seconds.

To conclude, CBAM provides the strongest mean primary detection score in our fixed YOLOv8n configuration. Its gain is modest and comes with a measurable latency cost. This supports it as a candidate for this dataset and placement, not as a universally best attention mechanism. Our main limitations are the small dataset, uncertain physical-fruit independence, three training seeds and the absence of healthy-image negatives. The next priorities are stronger identity-based splits, healthy negatives and equal-budget placement comparisons. Deployment work would also need complete camera-to-decision timing. The practical contribution is a reproducible comparison that shows both the benefit and the remaining errors. Thank you.

Delivery cue: Finish the spoken content at approximately 9 minutes. Advance to references without reading every entry.

# Rehearsal and recording checklist

- Both students must speak for 4-5 minutes. Practise with a timer and aim for 4:30 each.
- Explain terms in plain language: feature channels, spatial attention, IoU and mAP.
- Say percentage points for the 1.41 gain. Avoid claiming statistical significance or tested robustness.
- Open Presenter View to see the notes while presenting. Confirm the recording captures the slides, not the notes.
- Keep the microphone distance and recording volume consistent between presenters.
- Export an MP4 using your recording software. Watch the complete exported video, checking audio, slide changes and total duration of 8-10 minutes.
- Confirm the file plays correctly before uploading to Canvas by 25 October, 23:59 Auckland time.

# Questions to prepare for

1. Why this insertion point? It retains stride-16 spatial detail at manageable cost. We did not establish an optimal placement.
2. Why three seeds? They expose training variation within the course budget, but cannot establish statistical significance.
3. Why grouped splitting? Related captures could inflate performance across random partitions. The grouping remains heuristic.
4. Why CBAM? It has the highest mean mAP50-95 and positive paired gains at all three seeds, with higher latency.
5. Why no good/bad accuracy? The dataset and task use local defect boxes, not healthy-versus-defective image labels.
6. Does CBAM focus on defects? We did not evaluate attention maps or causal feature attribution.
7. Are results comparable to published SSDA numbers? Not directly, because our split and protocol differ.
8. Why no real-world FPS claim? Timing covers only a fused model forward pass, excluding preprocessing and NMS.
