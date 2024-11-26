Progressive Dual-Branch Diffusion Model: A Novel Approach for Robust 2D Human Pose Estimation
=
This is the code for this paper

Environment
=
```

conda create -n ProDiffPose python=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111 # install pytorch

```

Train
=
```

python main.py --config cfg_files/poseseg.yaml

```

Test
=
```

python main.py --config cfg_files/poseseg.yaml

```
Acknowledgements
=
Thanks for the open-source:

[Alphapose](https://github.com/MVIG-SJTU/AlphaPose)

[Pose2UV](https://github.com/boycehbz/Pose2UV?tab=readme-ov-file)
