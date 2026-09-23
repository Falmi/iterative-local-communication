# final_cifar100

Status: complete

Successful-run conditional summaries. Failure has no accuracy. C T1 is the one-shot control, no duplicate experiment. Small-seed unadjusted Student-t intervals; not equivalence tests.

Failures/pending:
```json
[
  {
    "id": "cifar-100-A-T0-seed1",
    "model": "A",
    "seed": 1,
    "T": 0,
    "dataset": "CIFAR-100",
    "status": "completed"
  },
  {
    "id": "cifar-100-A-T0-seed2",
    "model": "A",
    "seed": 2,
    "T": 0,
    "dataset": "CIFAR-100",
    "status": "completed"
  },
  {
    "id": "cifar-100-A-T0-seed3",
    "model": "A",
    "seed": 3,
    "T": 0,
    "dataset": "CIFAR-100",
    "status": "completed"
  },
  {
    "id": "cifar-100-C-T3-seed1",
    "model": "C",
    "seed": 1,
    "T": 3,
    "dataset": "CIFAR-100",
    "status": "failed",
    "failure": {
      "epoch": 28,
      "batch": 188,
      "global_step": 9692,
      "learning_rate": 0.09557016383177226,
      "stage": "gradient",
      "quantity": "backbone.0.weight",
      "loss": 2.0141359772949545e+19,
      "previous_loss": 7.895689206326886e+18
    }
  },
  {
    "id": "cifar-100-C-T3-seed2",
    "model": "C",
    "seed": 2,
    "T": 3,
    "dataset": "CIFAR-100",
    "status": "failed",
    "failure": {
      "epoch": 16,
      "batch": 81,
      "global_step": 5361,
      "learning_rate": 0.09861849601988384,
      "stage": "gradient",
      "quantity": "backbone.0.weight",
      "loss": 2.295201746701517e+16,
      "previous_loss": 4.6414728879354675e+17
    }
  },
  {
    "id": "cifar-100-C-T3-seed3",
    "model": "C",
    "seed": 3,
    "T": 3,
    "dataset": "CIFAR-100",
    "status": "completed"
  },
  {
    "id": "cifar-100-D1-T3-seed1",
    "model": "D1",
    "seed": 1,
    "T": 3,
    "dataset": "CIFAR-100",
    "status": "completed"
  },
  {
    "id": "cifar-100-D1-T3-seed2",
    "model": "D1",
    "seed": 2,
    "T": 3,
    "dataset": "CIFAR-100",
    "status": "completed"
  },
  {
    "id": "cifar-100-D1-T3-seed3",
    "model": "D1",
    "seed": 3,
    "T": 3,
    "dataset": "CIFAR-100",
    "status": "completed"
  }
]
```

Aggregates and paired comparisons:
```json
{
  "aggregates": {
    "A": {
      "seeds": [
        1,
        2,
        3
      ],
      "accuracy_pp": {
        "n": 3,
        "mean": 77.78,
        "std": 0.06999999999999826,
        "ci95": [
          77.60611036017964,
          77.95388963982036
        ],
        "median": 77.81,
        "minimum": 77.7,
        "maximum": 77.83
      },
      "NLL": {
        "n": 3,
        "mean": 0.905869240697225,
        "std": 0.006458723088771694,
        "ci95": [
          0.8898248831028535,
          0.9219135982915965
        ],
        "median": 0.9021940015792846,
        "minimum": 0.9020868453979493,
        "maximum": 0.9133268751144409
      },
      "ECE": {
        "n": 3,
        "mean": 0.04727956791271766,
        "std": 0.003006007427327829,
        "ci95": [
          0.03981223150078355,
          0.05474690432465177
        ],
        "median": 0.047573425992578265,
        "minimum": 0.04413742331713438,
        "maximum": 0.05012785442844033
      },
      "Brier": {
        "n": 3,
        "mean": 0.3183781093676885,
        "std": 0.0018263546393190175,
        "ci95": [
          0.3138411929331822,
          0.32291502580219483
        ],
        "median": 0.3192072696208954,
        "minimum": 0.31628426117897035,
        "maximum": 0.3196427973031998
      },
      "inference_latency_ms_per_sample": {
        "n": 3,
        "mean": 0.059547978339833205,
        "std": 0.00011049101540156104,
        "ci95": [
          0.059273503441668,
          0.05982245323799841
        ],
        "median": 0.05953358760161791,
        "minimum": 0.05944538780604489,
        "maximum": 0.05966495961183682
      }
    },
    "C": {
      "seeds": [
        3
      ],
      "accuracy_pp": {
        "n": 1,
        "mean": 77.34,
        "std": null,
        "ci95": null,
        "median": 77.34,
        "minimum": 77.34,
        "maximum": 77.34
      },
      "NLL": {
        "n": 1,
        "mean": 0.9191459184646606,
        "std": null,
        "ci95": null,
        "median": 0.9191459184646606,
        "minimum": 0.9191459184646606,
        "maximum": 0.9191459184646606
      },
      "ECE": {
        "n": 1,
        "mean": 0.04828338041193783,
        "std": null,
        "ci95": null,
        "median": 0.04828338041193783,
        "minimum": 0.04828338041193783,
        "maximum": 0.04828338041193783
      },
      "Brier": {
        "n": 1,
        "mean": 0.32259983024597166,
        "std": null,
        "ci95": null,
        "median": 0.32259983024597166,
        "minimum": 0.32259983024597166,
        "maximum": 0.32259983024597166
      },
      "inference_latency_ms_per_sample": {
        "n": 1,
        "mean": 0.06388916796422564,
        "std": null,
        "ci95": null,
        "median": 0.06388916796422564,
        "minimum": 0.06388916796422564,
        "maximum": 0.06388916796422564
      }
    },
    "D1": {
      "seeds": [
        1,
        2,
        3
      ],
      "accuracy_pp": {
        "n": 3,
        "mean": 76.68,
        "std": 0.605557594288111,
        "ci95": [
          75.17571154341074,
          78.18428845658927
        ],
        "median": 76.71,
        "minimum": 76.06,
        "maximum": 77.27000000000001
      },
      "NLL": {
        "n": 3,
        "mean": 1.1819996015230814,
        "std": 0.031535382901885387,
        "ci95": [
          1.103661367602992,
          1.2603378354431707
        ],
        "median": 1.1814856852531432,
        "minimum": 1.1507243175506592,
        "maximum": 1.213788801765442
      },
      "ECE": {
        "n": 3,
        "mean": 0.1589781825090448,
        "std": 0.005398079464699978,
        "ci95": [
          0.14556860973992472,
          0.17238775527816488
        ],
        "median": 0.15678030689954758,
        "minimum": 0.15502575722932815,
        "maximum": 0.1651284833982587
      },
      "Brier": {
        "n": 3,
        "mean": 0.38265247325897217,
        "std": 0.008297349644889878,
        "ci95": [
          0.3620407140987784,
          0.4032642324191659
        ],
        "median": 0.38103888459205626,
        "minimum": 0.37528043761253355,
        "maximum": 0.39163809757232665
      },
      "inference_latency_ms_per_sample": {
        "n": 3,
        "mean": 0.06546830107496741,
        "std": 1.1368558425158698e-05,
        "ci95": [
          0.0654400600102556,
          0.06549654213967923
        ],
        "median": 0.06546281001647003,
        "minimum": 0.06546072040800936,
        "maximum": 0.06548137280042284
      }
    }
  },
  "paired_comparisons": {
    "C-A": {
      "n": 1,
      "mean": -0.49000000000000155,
      "std": null,
      "ci95": null,
      "median": -0.49000000000000155,
      "minimum": -0.49000000000000155,
      "maximum": -0.49000000000000155,
      "individual_differences": [
        {
          "seed": 3,
          "difference_pp": -0.49000000000000155
        }
      ],
      "paired_t_test": {
        "statistic": null,
        "pvalue": null,
        "reason": "insufficient pairs or zero variance"
      },
      "positive_seeds": 0,
      "negative_seeds": 1,
      "zero_seeds": 0,
      "standardized_paired_effect": null
    },
    "D1-A": {
      "n": 3,
      "mean": -1.0999999999999972,
      "std": 0.6223343153000627,
      "ci95": [
        -2.645964141934045,
        0.44596414193405076
      ],
      "median": -0.990000000000002,
      "minimum": -1.7699999999999938,
      "maximum": -0.539999999999996,
      "individual_differences": [
        {
          "seed": 1,
          "difference_pp": -0.990000000000002
        },
        {
          "seed": 2,
          "difference_pp": -0.539999999999996
        },
        {
          "seed": 3,
          "difference_pp": -1.7699999999999938
        }
      ],
      "paired_t_test": {
        "statistic": -3.061466869952572,
        "pvalue": 0.09217931243948904,
        "reason": null
      },
      "positive_seeds": 0,
      "negative_seeds": 3,
      "zero_seeds": 0,
      "standardized_paired_effect": -1.7675387214822387
    },
    "D1-C": {
      "n": 1,
      "mean": -1.2799999999999923,
      "std": null,
      "ci95": null,
      "median": -1.2799999999999923,
      "minimum": -1.2799999999999923,
      "maximum": -1.2799999999999923,
      "individual_differences": [
        {
          "seed": 3,
          "difference_pp": -1.2799999999999923
        }
      ],
      "paired_t_test": {
        "statistic": null,
        "pvalue": null,
        "reason": "insufficient pairs or zero variance"
      },
      "positive_seeds": 0,
      "negative_seeds": 1,
      "zero_seeds": 0,
      "standardized_paired_effect": null
    }
  }
}
```
