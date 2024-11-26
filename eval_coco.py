from alphapose_module.alphapose.utils.metrics import evaluate_mAP


res = evaluate_mAP('output/test/06.13-13h21m00s/validate_kpt_epoch0000.json', ann_type='keypoints', ann_file='data/person_keypoints_val2017.json', halpe=None)

print(res)