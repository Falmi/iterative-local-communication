# final_iteration_ablation

Status: complete

Successful-run conditional summaries. Failure has no accuracy. C T1 is the one-shot control, no duplicate experiment. Small-seed unadjusted Student-t intervals; not equivalence tests.

Failures/pending:
```json
[
  {
    "id": "cifar-10-C-T1-seed1",
    "model": "C",
    "seed": 1,
    "T": 1,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-C-T1-seed2",
    "model": "C",
    "seed": 2,
    "T": 1,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-C-T1-seed3",
    "model": "C",
    "seed": 3,
    "T": 1,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-C-T2-seed1",
    "model": "C",
    "seed": 1,
    "T": 2,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-C-T2-seed2",
    "model": "C",
    "seed": 2,
    "T": 2,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-C-T2-seed3",
    "model": "C",
    "seed": 3,
    "T": 2,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-C-T5-seed1",
    "model": "C",
    "seed": 1,
    "T": 5,
    "dataset": "CIFAR-10",
    "status": "failed",
    "failure": {
      "epoch": 15,
      "phase": "validation",
      "evaluation_batch": 1,
      "last_completed_training_batch": 352,
      "global_step": 5280,
      "learning_rate": 0.09879583809693739,
      "stage": "validation_forward",
      "quantity": "backbone.6.1.bn2",
      "loss": null,
      "previous_loss": null,
      "evidence_origin": "Isolated replay of uncommitted epoch 15 from original epoch-14 checkpoint",
      "evidence_files": {
        "work/final_validation_diagnostics/replay_epoch15/event.json": "32cdb26ad076bce16babc9c95ec6e7bc39b9a22b0aa320d02a27469d83dd8f76",
        "work/final_validation_diagnostics/replay_epoch15/model_at_event.pt": "7b4efc1d1f5d663da3a71030e8b411bf4609818786a07d6e4737037c8a4bee85",
        "work/final_validation_diagnostics/replay_epoch15/last.pt": "2a1f9330e37682a9a42d8d1a4a167f9995387fcc397d4ba40338e3b0d51f3f14",
        "work/final_validation_diagnostics/replay_epoch15/replay_result.json": "f44b5968fa9809aab8bcaef6a59540e19e45ac3e27f22c60e044cbe0cf3ce4db"
      },
      "inf_count": 7418,
      "nan_count": 0
    }
  },
  {
    "id": "cifar-10-C-T5-seed2",
    "model": "C",
    "seed": 2,
    "T": 5,
    "dataset": "CIFAR-10",
    "status": "failed",
    "failure": {
      "epoch": 1,
      "batch": 47,
      "global_step": 47,
      "learning_rate": 0.1,
      "stage": "gradient",
      "quantity": "backbone.0.weight",
      "loss": 1.2909460139913354e+20,
      "previous_loss": 5.713643839672864e+19
    }
  },
  {
    "id": "cifar-10-C-T5-seed3",
    "model": "C",
    "seed": 3,
    "T": 5,
    "dataset": "CIFAR-10",
    "status": "failed",
    "failure": {
      "epoch": 1,
      "batch": 64,
      "global_step": 64,
      "learning_rate": 0.1,
      "stage": "gradient",
      "quantity": "backbone.0.weight",
      "loss": 8.0591906535702e+19,
      "previous_loss": 8.00216734172583e+19
    }
  },
  {
    "id": "cifar-10-D1-T1-seed1",
    "model": "D1",
    "seed": 1,
    "T": 1,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-D1-T1-seed2",
    "model": "D1",
    "seed": 2,
    "T": 1,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-D1-T1-seed3",
    "model": "D1",
    "seed": 3,
    "T": 1,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-D1-T2-seed1",
    "model": "D1",
    "seed": 1,
    "T": 2,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-D1-T2-seed2",
    "model": "D1",
    "seed": 2,
    "T": 2,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-D1-T2-seed3",
    "model": "D1",
    "seed": 3,
    "T": 2,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-D1-T5-seed1",
    "model": "D1",
    "seed": 1,
    "T": 5,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-D1-T5-seed2",
    "model": "D1",
    "seed": 2,
    "T": 5,
    "dataset": "CIFAR-10",
    "status": "completed"
  },
  {
    "id": "cifar-10-D1-T5-seed3",
    "model": "D1",
    "seed": 3,
    "T": 5,
    "dataset": "CIFAR-10",
    "status": "completed"
  }
]
```

Aggregates and paired comparisons:
```json
{
  "aggregates": {
    "C-T1": {
      "seeds": [
        1,
        2,
        3
      ],
      "accuracy_pp": {
        "n": 3,
        "mean": 94.88333333333334,
        "std": 0.20792626898334426,
        "ci95": [
          94.36681584729467,
          95.399850819372
        ],
        "median": 94.8,
        "minimum": 94.73,
        "maximum": 95.12
      },
      "NLL": {
        "n": 3,
        "mean": 0.22760122546851635,
        "std": 0.00916642850718503,
        "ci95": [
          0.20483055473203693,
          0.2503718962049958
        ],
        "median": 0.23190460596084594,
        "minimum": 0.2170749248534441,
        "maximum": 0.233824145591259
      },
      "ECE": {
        "n": 3,
        "mean": 0.03580810434321562,
        "std": 0.001749098460656336,
        "ci95": [
          0.03146310289558861,
          0.040153105790842636
        ],
        "median": 0.03657766685783863,
        "minimum": 0.033806172636151315,
        "maximum": 0.037040473535656926
      },
      "Brier": {
        "n": 3,
        "mean": 0.08602299581145247,
        "std": 0.003067618474834229,
        "ci95": [
          0.07840260907294916,
          0.09364338254995577
        ],
        "median": 0.08747493287324905,
        "minimum": 0.08249895791336893,
        "maximum": 0.08809509664773942
      },
      "inference_latency_ms_per_sample": {
        "n": 3,
        "mean": 0.061131647787988186,
        "std": 1.5295148233018686e-05,
        "ci95": [
          0.0610936525334562,
          0.06116964304252017
        ],
        "median": 0.061130921391304584,
        "minimum": 0.0611167287803255,
        "maximum": 0.06114729319233447
      }
    },
    "C-T2": {
      "seeds": [
        1,
        2,
        3
      ],
      "accuracy_pp": {
        "n": 3,
        "mean": 95.04666666666667,
        "std": 0.06429100507328683,
        "ci95": [
          94.88695895643976,
          95.20637437689358
        ],
        "median": 95.02000000000001,
        "minimum": 95.0,
        "maximum": 95.12
      },
      "NLL": {
        "n": 3,
        "mean": 0.21311809724966685,
        "std": 0.02367734410024314,
        "ci95": [
          0.15430031385689258,
          0.2719358806424411
        ],
        "median": 0.20880717306137084,
        "minimum": 0.1918924008369446,
        "maximum": 0.23865471785068512
      },
      "ECE": {
        "n": 3,
        "mean": 0.03184125501513481,
        "std": 0.003993753141106814,
        "ci95": [
          0.02192022222601298,
          0.04176228780425664
        ],
        "median": 0.03202692560851574,
        "minimum": 0.027757904842495917,
        "maximum": 0.03573893459439278
      },
      "Brier": {
        "n": 3,
        "mean": 0.08184529449542363,
        "std": 0.0034492841046964116,
        "ci95": [
          0.07327679777251249,
          0.09041379121833477
        ],
        "median": 0.08094443757534027,
        "minimum": 0.07893582661151886,
        "maximum": 0.08565561929941178
      },
      "inference_latency_ms_per_sample": {
        "n": 3,
        "mean": 0.06250948806797775,
        "std": 4.0364383352023345e-05,
        "ci95": [
          0.06240921738108268,
          0.06260975875487282
        ],
        "median": 0.06251127778668888,
        "minimum": 0.06246825859416276,
        "maximum": 0.06254892782308161
      }
    },
    "C-T3": {
      "seeds": [
        1,
        2,
        3
      ],
      "accuracy_pp": {
        "n": 3,
        "mean": 95.15,
        "std": 0.11532562594670781,
        "ci95": [
          94.86351526345813,
          95.43648473654189
        ],
        "median": 95.11,
        "minimum": 95.06,
        "maximum": 95.28
      },
      "NLL": {
        "n": 3,
        "mean": 0.1904127191821734,
        "std": 0.002728153691922181,
        "ci95": [
          0.1836356097127026,
          0.1971898286516442
        ],
        "median": 0.19013547785282134,
        "minimum": 0.18783377190828324,
        "maximum": 0.19326890778541564
      },
      "ECE": {
        "n": 3,
        "mean": 0.0277351985608538,
        "std": 0.0018429078381331239,
        "ci95": [
          0.023157161700923766,
          0.032313235420783834
        ],
        "median": 0.027868011732399462,
        "minimum": 0.025829476940631868,
        "maximum": 0.029508107009530066
      },
      "Brier": {
        "n": 3,
        "mean": 0.07871206740736962,
        "std": 0.0005823898803643963,
        "ci95": [
          0.07726533074263259,
          0.08015880407210665
        ],
        "median": 0.07904054374694824,
        "minimum": 0.07803964084386826,
        "maximum": 0.07905601763129234
      },
      "inference_latency_ms_per_sample": {
        "n": 3,
        "mean": 0.06372679187140116,
        "std": 2.6280448622773325e-05,
        "ci95": [
          0.06366150761789642,
          0.06379207612490591
        ],
        "median": 0.06371854839962907,
        "minimum": 0.06370562139782124,
        "maximum": 0.0637562058167532
      }
    },
    "D1-T1": {
      "seeds": [
        1,
        2,
        3
      ],
      "accuracy_pp": {
        "n": 3,
        "mean": 95.08333333333333,
        "std": 0.08504900548115472,
        "ci95": [
          94.87205989147336,
          95.2946067751933
        ],
        "median": 95.08,
        "minimum": 95.0,
        "maximum": 95.17
      },
      "NLL": {
        "n": 3,
        "mean": 0.18789219016631445,
        "std": 0.0065875960779577525,
        "ci95": [
          0.17152769431948384,
          0.20425668601314506
        ],
        "median": 0.18484898953437806,
        "minimum": 0.18337635563611984,
        "maximum": 0.19545122532844544
      },
      "ECE": {
        "n": 3,
        "mean": 0.029020490513245265,
        "std": 0.0014798412263415386,
        "ci95": [
          0.02534436111553295,
          0.03269661991095758
        ],
        "median": 0.02908565405011177,
        "minimum": 0.027509143942594527,
        "maximum": 0.030466673547029494
      },
      "Brier": {
        "n": 3,
        "mean": 0.07982747633854548,
        "std": 0.001787961937218574,
        "ci95": [
          0.07538593266318168,
          0.08426902001390928
        ],
        "median": 0.07946388982534408,
        "minimum": 0.07824925211668014,
        "maximum": 0.08176928707361221
      },
      "inference_latency_ms_per_sample": {
        "n": 3,
        "mean": 0.06420957527977104,
        "std": 0.0017381266679632361,
        "ci95": [
          0.059891829276138134,
          0.06852732128340396
        ],
        "median": 0.06352834122953936,
        "minimum": 0.0629152536392212,
        "maximum": 0.06618513097055256
      }
    },
    "D1-T2": {
      "seeds": [
        1,
        2,
        3
      ],
      "accuracy_pp": {
        "n": 3,
        "mean": 95.17999999999999,
        "std": 0.11135528725659997,
        "ci95": [
          94.90337813152651,
          95.45662186847348
        ],
        "median": 95.16,
        "minimum": 95.08,
        "maximum": 95.3
      },
      "NLL": {
        "n": 3,
        "mean": 0.18861464144388834,
        "std": 0.0011622872321593405,
        "ci95": [
          0.1857273598986312,
          0.19150192298914548
        ],
        "median": 0.18837296028137207,
        "minimum": 0.18759219541549682,
        "maximum": 0.18987876863479614
      },
      "ECE": {
        "n": 3,
        "mean": 0.02835357171992461,
        "std": 0.00024704947379392336,
        "ci95": [
          0.027739866805412655,
          0.028967276634436566
        ],
        "median": 0.028362364745140076,
        "minimum": 0.02810224312245846,
        "maximum": 0.028596107292175293
      },
      "Brier": {
        "n": 3,
        "mean": 0.0794515408039093,
        "std": 0.0010340740768643255,
        "ci95": [
          0.07688275839285906,
          0.08202032321495954
        ],
        "median": 0.07976406036615372,
        "minimum": 0.07829725404977798,
        "maximum": 0.0802933079957962
      },
      "inference_latency_ms_per_sample": {
        "n": 3,
        "mean": 0.06394295860858014,
        "std": 0.002267674098684806,
        "ci95": [
          0.05830974386214758,
          0.06957617335501269
        ],
        "median": 0.06292164246551693,
        "minimum": 0.062365547998342666,
        "maximum": 0.0665416853618808
      }
    },
    "D1-T3": {
      "seeds": [
        1,
        2,
        3
      ],
      "accuracy_pp": {
        "n": 3,
        "mean": 95.01666666666667,
        "std": 0.08504900548114777,
        "ci95": [
          94.8053932248067,
          95.22794010852662
        ],
        "median": 95.02000000000001,
        "minimum": 94.93,
        "maximum": 95.1
      },
      "NLL": {
        "n": 3,
        "mean": 0.19078922318617503,
        "std": 0.006540791472418572,
        "ci95": [
          0.17454099642504645,
          0.2070374499473036
        ],
        "median": 0.19178413202762604,
        "minimum": 0.1838079758644104,
        "maximum": 0.19677556166648866
      },
      "ECE": {
        "n": 3,
        "mean": 0.029951370812455812,
        "std": 0.0008408457055279665,
        "ci95": [
          0.02786259428561636,
          0.032040147339295263
        ],
        "median": 0.0296220624178648,
        "minimum": 0.02932502027750015,
        "maximum": 0.030907029742002486
      },
      "Brier": {
        "n": 3,
        "mean": 0.08031002373298009,
        "std": 0.001774305743723487,
        "ci95": [
          0.07590240392287598,
          0.0847176435430842
        ],
        "median": 0.07995253794193267,
        "minimum": 0.07874167939424515,
        "maximum": 0.08223585386276246
      },
      "inference_latency_ms_per_sample": {
        "n": 3,
        "mean": 0.06541824873226385,
        "std": 0.0002566050024870972,
        "ci95": [
          0.06478080656856976,
          0.06605569089595793
        ],
        "median": 0.06528483820147812,
        "minimum": 0.0652558309957385,
        "maximum": 0.0657140769995749
      }
    },
    "D1-T5": {
      "seeds": [
        1,
        2,
        3
      ],
      "accuracy_pp": {
        "n": 3,
        "mean": 91.98,
        "std": 4.591862367275393,
        "ci95": [
          80.57318152642542,
          103.38681847357459
        ],
        "median": 94.19999999999999,
        "minimum": 86.7,
        "maximum": 95.04
      },
      "NLL": {
        "n": 3,
        "mean": 0.276550542596976,
        "std": 0.10263791845826302,
        "ci95": [
          0.021583818702409163,
          0.5315172664915429
        ],
        "median": 0.2430420494556427,
        "minimum": 0.19485465980768205,
        "maximum": 0.39175491852760314
      },
      "ECE": {
        "n": 3,
        "mean": 0.028112567252417404,
        "std": 0.009325196230768664,
        "ci95": [
          0.004947495626380003,
          0.05127763887845481
        ],
        "median": 0.03165091363191604,
        "minimum": 0.017536046935617923,
        "maximum": 0.03515074118971825
      },
      "Brier": {
        "n": 3,
        "mean": 0.12150095771153768,
        "std": 0.060347194341091794,
        "ci95": [
          -0.02840978354763679,
          0.27141169897071216
        ],
        "median": 0.09369821412563324,
        "minimum": 0.08006663169860839,
        "maximum": 0.1907380273103714
      },
      "inference_latency_ms_per_sample": {
        "n": 3,
        "mean": 0.08233056479754547,
        "std": 0.0015112934587618368,
        "ci95": [
          0.07857630372316013,
          0.08608482587193081
        ],
        "median": 0.08216712818248198,
        "minimum": 0.08090763222426176,
        "maximum": 0.08391693398589269
      }
    }
  },
  "paired_comparisons": {
    "C-T1-C-T3": {
      "n": 3,
      "mean": -0.2666666666666632,
      "std": 0.11015141094572066,
      "ci95": [
        -0.5402979405960451,
        0.006964607262718636
      ],
      "median": -0.26000000000000467,
      "minimum": -0.37999999999999146,
      "maximum": -0.15999999999999348,
      "individual_differences": [
        {
          "seed": 1,
          "difference_pp": -0.15999999999999348
        },
        {
          "seed": 2,
          "difference_pp": -0.26000000000000467
        },
        {
          "seed": 3,
          "difference_pp": -0.37999999999999146
        }
      ],
      "paired_t_test": {
        "statistic": -4.193139346887671,
        "pvalue": 0.05244126064173145,
        "reason": null
      },
      "positive_seeds": 0,
      "negative_seeds": 3,
      "zero_seeds": 0,
      "standardized_paired_effect": -2.4209101306752085
    },
    "C-T2-C-T3": {
      "n": 3,
      "mean": -0.10333333333332935,
      "std": 0.17039170558843125,
      "ci95": [
        -0.5266097949497655,
        0.3199431282831068
      ],
      "median": -0.08999999999999009,
      "minimum": -0.28000000000000247,
      "maximum": 0.060000000000004494,
      "individual_differences": [
        {
          "seed": 1,
          "difference_pp": -0.28000000000000247
        },
        {
          "seed": 2,
          "difference_pp": 0.060000000000004494
        },
        {
          "seed": 3,
          "difference_pp": -0.08999999999999009
        }
      ],
      "paired_t_test": {
        "statistic": -1.0503949287360665,
        "pvalue": 0.4037358877125877,
        "reason": null
      },
      "positive_seeds": 1,
      "negative_seeds": 2,
      "zero_seeds": 0,
      "standardized_paired_effect": -0.6064457948611858
    },
    "C-T5-C-T3": {
      "n": 0,
      "mean": null,
      "std": null,
      "ci95": null,
      "median": null,
      "minimum": null,
      "maximum": null,
      "individual_differences": [],
      "paired_t_test": {
        "statistic": null,
        "pvalue": null,
        "reason": "insufficient pairs or zero variance"
      },
      "positive_seeds": 0,
      "negative_seeds": 0,
      "zero_seeds": 0,
      "standardized_paired_effect": null
    },
    "D1-T1-D1-T3": {
      "n": 3,
      "mean": 0.06666666666666303,
      "std": 0.0850490054811499,
      "ci95": [
        -0.14460677519330378,
        0.2779401085266298
      ],
      "median": 0.06999999999999229,
      "minimum": -0.019999999999997797,
      "maximum": 0.14999999999999458,
      "individual_differences": [
        {
          "seed": 1,
          "difference_pp": -0.019999999999997797
        },
        {
          "seed": 2,
          "difference_pp": 0.06999999999999229
        },
        {
          "seed": 3,
          "difference_pp": 0.14999999999999458
        }
      ],
      "paired_t_test": {
        "statistic": 1.3576884666042501,
        "pvalue": 0.3074566389355593,
        "reason": null
      },
      "positive_seeds": 2,
      "negative_seeds": 1,
      "zero_seeds": 0,
      "standardized_paired_effect": 0.783861801669614
    },
    "D1-T2-D1-T3": {
      "n": 3,
      "mean": 0.16333333333333014,
      "std": 0.09073771725877731,
      "ci95": [
        -0.06207165198454409,
        0.38873831865120434
      ],
      "median": 0.20000000000000018,
      "minimum": 0.05999999999999339,
      "maximum": 0.22999999999999687,
      "individual_differences": [
        {
          "seed": 1,
          "difference_pp": 0.20000000000000018
        },
        {
          "seed": 2,
          "difference_pp": 0.22999999999999687
        },
        {
          "seed": 3,
          "difference_pp": 0.05999999999999339
        }
      ],
      "paired_t_test": {
        "statistic": 3.117795338581159,
        "pvalue": 0.08930742443742982,
        "reason": null
      },
      "positive_seeds": 3,
      "negative_seeds": 0,
      "zero_seeds": 0,
      "standardized_paired_effect": 1.8000599780079924
    },
    "D1-T5-D1-T3": {
      "n": 3,
      "mean": -3.0366666666666693,
      "std": 4.6680009997142555,
      "ci95": [
        -14.632623988401392,
        8.559290655068054
      ],
      "median": -0.8200000000000096,
      "minimum": -8.399999999999997,
      "maximum": 0.10999999999999899,
      "individual_differences": [
        {
          "seed": 1,
          "difference_pp": -8.399999999999997
        },
        {
          "seed": 2,
          "difference_pp": 0.10999999999999899
        },
        {
          "seed": 3,
          "difference_pp": -0.8200000000000096
        }
      ],
      "paired_t_test": {
        "statistic": -1.12674803468111,
        "pvalue": 0.3768650962152781,
        "reason": null
      },
      "positive_seeds": 1,
      "negative_seeds": 2,
      "zero_seeds": 0,
      "standardized_paired_effect": -0.6505282811320208
    }
  }
}
```
