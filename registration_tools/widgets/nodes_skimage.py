from skimage import data, draw, exposure, feature, filters, morphology, restoration, segmentation, transform
from numpy import uint8, uint32, int8, int16, int32, float16, float32, float64

NODES_SKIMAGE = {
  "data": {
    "skimage.data.astronaut": {
      "name": "astronaut",
      "function_name": "astronaut",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.astronaut,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut",
      "parameters": {}
    },
    "skimage.data.brain": {
      "name": "brain",
      "function_name": "brain",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.brain,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.brain",
      "parameters": {}
    },
    "skimage.data.brick": {
      "name": "brick",
      "function_name": "brick",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.brick,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.brick",
      "parameters": {}
    },
    "skimage.data.camera": {
      "name": "camera",
      "function_name": "camera",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.camera,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.camera",
      "parameters": {}
    },
    "skimage.data.cat": {
      "name": "cat",
      "function_name": "cat",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.cat,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.cat",
      "parameters": {}
    },
    "skimage.data.cell": {
      "name": "cell",
      "function_name": "cell",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.cell,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.cell",
      "parameters": {}
    },
    "skimage.data.cells3d": {
      "name": "cells3d",
      "function_name": "cells3d",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.cells3d,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.cells3d",
      "parameters": {}
    },
    "skimage.data.checkerboard": {
      "name": "checkerboard",
      "function_name": "checkerboard",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.checkerboard,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.checkerboard",
      "parameters": {}
    },
    "skimage.data.chelsea": {
      "name": "chelsea",
      "function_name": "chelsea",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.chelsea,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.chelsea",
      "parameters": {}
    },
    "skimage.data.clock": {
      "name": "clock",
      "function_name": "clock",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.clock,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.clock",
      "parameters": {}
    },
    "skimage.data.coffee": {
      "name": "coffee",
      "function_name": "coffee",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.coffee,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee",
      "parameters": {}
    },
    "skimage.data.coins": {
      "name": "coins",
      "function_name": "coins",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.coins,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coins",
      "parameters": {}
    },
    "skimage.data.colorwheel": {
      "name": "colorwheel",
      "function_name": "colorwheel",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.colorwheel,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.colorwheel",
      "parameters": {}
    },
    "skimage.data.eagle": {
      "name": "eagle",
      "function_name": "eagle",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.eagle,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.eagle",
      "parameters": {}
    },
    "skimage.data.grass": {
      "name": "grass",
      "function_name": "grass",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.grass,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.grass",
      "parameters": {}
    },
    "skimage.data.gravel": {
      "name": "gravel",
      "function_name": "gravel",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.gravel,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.gravel",
      "parameters": {}
    },
    "skimage.data.horse": {
      "name": "horse",
      "function_name": "horse",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.horse,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.horse",
      "parameters": {}
    },
    "skimage.data.hubble_deep_field": {
      "name": "hubble_deep_field",
      "function_name": "hubble_deep_field",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.hubble_deep_field,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.hubble_deep_field",
      "parameters": {}
    },
    "skimage.data.human_mitosis": {
      "name": "human_mitosis",
      "function_name": "human_mitosis",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.human_mitosis,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.human_mitosis",
      "parameters": {}
    },
    "skimage.data.immunohistochemistry": {
      "name": "immunohistochemistry",
      "function_name": "immunohistochemistry",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.immunohistochemistry,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.immunohistochemistry",
      "parameters": {}
    },
    "skimage.data.kidney": {
      "name": "kidney",
      "function_name": "kidney",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.kidney,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.kidney",
      "parameters": {}
    },
    # "lbp_frontal_face_cascade_filename": {
    #   "name": "lbp_frontal_face_cascade_filename",
    #   "function_name": "lbp_frontal_face_cascade_filename",
    #   "type": "image",
    #   "outputs": {
    #     "0": {
    #       "type": str,
    #       "name": "result"
    #     }
    #   },
    #   "function": data.lbp_frontal_face_cascade_filename,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.lbp_frontal_face_cascade_filename",
    #   "parameters": {}
    # },
    "skimage.data.lfw_subset": {
      "name": "lfw_subset",
      "function_name": "lfw_subset",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.lfw_subset,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.lfw_subset",
      "parameters": {}
    },
    "skimage.data.lily": {
      "name": "lily",
      "function_name": "lily",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.lily,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.lily",
      "parameters": {}
    },
    "skimage.data.logo": {
      "name": "logo",
      "function_name": "logo",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.logo,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.logo",
      "parameters": {}
    },
    "skimage.data.microaneurysms": {
      "name": "microaneurysms",
      "function_name": "microaneurysms",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.microaneurysms,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.microaneurysms",
      "parameters": {}
    },
    "skimage.data.moon": {
      "name": "moon",
      "function_name": "moon",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.moon,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.moon",
      "parameters": {}
    },
    "skimage.data.nickel_solidification": {
      "name": "nickel_solidification",
      "function_name": "nickel_solidification",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.nickel_solidification,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.nickel_solidification",
      "parameters": {}
    },
    "skimage.data.page": {
      "name": "page",
      "function_name": "page",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.page,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.page",
      "parameters": {}
    },
    "skimage.data.palisades_of_vogt": {
      "name": "palisades_of_vogt",
      "function_name": "palisades_of_vogt",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.palisades_of_vogt,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.palisades_of_vogt",
      "parameters": {}
    },
    "skimage.data.protein_transport": {
      "name": "protein_transport",
      "function_name": "protein_transport",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.protein_transport,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.protein_transport",
      "parameters": {}
    },
    "skimage.data.retina": {
      "name": "retina",
      "function_name": "retina",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.retina,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.retina",
      "parameters": {}
    },
    "skimage.data.rocket": {
      "name": "rocket",
      "function_name": "rocket",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.rocket,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.rocket",
      "parameters": {}
    },
    "skimage.data.shepp_logan_phantom": {
      "name": "shepp_logan_phantom",
      "function_name": "shepp_logan_phantom",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.shepp_logan_phantom,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.shepp_logan_phantom",
      "parameters": {}
    },
    "skimage.data.skin": {
      "name": "skin",
      "function_name": "skin",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.skin,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.skin",
      "parameters": {}
    },
    # "stereo_motorcycle": {
    #   "name": "stereo_motorcycle",
    #   "function_name": "stereo_motorcycle",
    #   "type": "image",
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     },
    #     "1": {
    #       "type": "image",
    #       "name": "image"
    #     },
    #     "2": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": data.stereo_motorcycle,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.stereo_motorcycle",
    #   "parameters": {}
    # },
    "skimage.data.text": {
      "name": "text",
      "function_name": "text",
      "type": "image",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": data.text,
      "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.text",
      "parameters": {}
    },
    # "vortex": {
    #   "name": "vortex",
    #   "function_name": "vortex",
    #   "type": "image",
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     },
    #     "1": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": data.vortex,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.vortex",
    #   "parameters": {}
    # }
  },
  "draw": {
    "skimage.draw.bezier_curve": {
      "name": "bezier_curve",
      "function_name": "bezier_curve",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.bezier_curve,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.bezier_curve",
      "parameters": {
        "r0": {
          "type": int,
          "default": 0
        },
        "c0": {
          "type": int,
          "default": 0
        },
        "r1": {
          "type": int,
          "default": 1
        },
        "c1": {
          "type": int,
          "default": 1
        },
        "r2": {
          "type": int,
          "default": 2
        },
        "c2": {
          "type": int,
          "default": 2
        },
        "weight": {
          "type": float,
          "default": 1
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        }
      }
    },
    "skimage.draw.circle_perimeter": {
      "name": "circle_perimeter",
      "function_name": "circle_perimeter",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.circle_perimeter,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.circle_perimeter",
      "parameters": {
        "r": {
          "type": int,
          "default": 0
        },
        "c": {
          "type": int,
          "default": 0
        },
        "radius": {
          "type": int,
          "default": 1
        },
        "method": {
          "type": "str",
          "default": "bresenham"
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        }
      }
    },
    "skimage.draw.circle_perimeter_aa": {
      "name": "circle_perimeter_aa",
      "function_name": "circle_perimeter_aa",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc, val"
        }
      },
      "function": draw.circle_perimeter_aa,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.circle_perimeter_aa",
      "parameters": {
        "r": {
          "type": int,
          "default": 0
        },
        "c": {
          "type": int,
          "default": 0
        },
        "radius": {
          "type": int,
          "default": 1
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        }
      }
    },
    "skimage.draw.disk": {
      "name": "disk",
      "function_name": "disk",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.disk,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.disk",
      "parameters": {
        "center": {
          "type": tuple,
          "default": None
        },
        "radius": {
          "type": int,
          "default": 1
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        }
      }
    },
    "skimage.draw.ellipse": {
      "name": "ellipse",
      "function_name": "ellipse",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.ellipse,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.ellipse",
      "parameters": {
        "r": {
          "type": int,
          "default": 0
        },
        "c": {
          "type": int,
          "default": 0
        },
        "r_radius": {
          "type": int,
          "default": 1
        },
        "c_radius": {
          "type": int,
          "default": 1
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        },
        "rotation": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.draw.ellipse_perimeter": {
      "name": "ellipse_perimeter",
      "function_name": "ellipse_perimeter",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.ellipse_perimeter,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.ellipse_perimeter",
      "parameters": {
        "r": {
          "type": int,
          "default": 0
        },
        "c": {
          "type": int,
          "default": 0
        },
        "r_radius": {
          "type": int,
          "default": 1
        },
        "c_radius": {
          "type": int,
          "default": 1
        },
        "orientation": {
          "type": int,
          "default": 0
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        }
      }
    },
    "skimage.draw.ellipsoid": {
      "name": "ellipsoid",
      "function_name": "ellipsoid",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc, zz"
        }
      },
      "function": draw.ellipsoid,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.ellipsoid",
      "parameters": {
        "a": {
          "type": int,
          "default": 0
        },
        "b": {
          "type": int,
          "default": 0
        },
        "c": {
          "type": int,
          "default": 0
        },
        "spacing": {
          "type": tuple,
          "default": (1.0,1.0,1.0)
        },
        "levelset": {
          "type": bool,
          "default": False
        }
      }
    },
    "skimage.draw.ellipsoid_stats": {
      "name": "ellipsoid_stats",
      "function_name": "ellipsoid_stats",
      "type": "function",
      "outputs": {
        "0": {
          "type": float,
          "name": "volume"
        },
        "1": {
          "type": float,
          "name": "area"
        }
      },
      "function": draw.ellipsoid_stats,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.ellipsoid_stats",
      "parameters": {
        "a": {
          "type": int,
          "default": 0
        },
        "b": {
          "type": int,
          "default": 0
        },
        "c": {
          "type": int,
          "default": 0
        }
      }
    },
    "skimage.draw.line": {
      "name": "line",
      "function_name": "line",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.line,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.line",
      "parameters": {
        "r0": {
          "type": int,
          "default": 0
        },
        "c0": {
          "type": int,
          "default": 0
        },
        "r1": {
          "type": int,
          "default": 1
        },
        "c1": {
          "type": int,
          "default": 1
        }
      }
    },
    "skimage.draw.line_aa": {
      "name": "line_aa",
      "function_name": "line_aa",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc, val"
        }
      },
      "function": draw.line_aa,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.line_aa",
      "parameters": {
        "r0": {
          "type": int,
          "default": 0
        },
        "c0": {
          "type": int,
          "default": 0
        },
        "r1": {
          "type": int,
          "default": 1
        },
        "c1": {
          "type": int,
          "default": 1
        }
      }
    },
    "skimage.draw.line_nd": {
      "name": "line_nd",
      "function_name": "line_nd",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates_nd",
          "name": "rr, cc"
        }
      },
      "function": draw.line_nd,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.line_nd",
      "parameters": {
        "start": {
          "type": tuple,
          "default": (0,0,0)
        },
        "stop": {
          "type": tuple,
          "default": (0,0,0)
        },
        "endpoint": {
          "type": bool,
          "default": False
        },
        "integer": {
          "type": bool,
          "default": True
        }
      }
    },
    "skimage.draw.polygon": {
      "name": "polygon",
      "function_name": "polygon",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.polygon,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.polygon",
      "parameters": {
        "r": {
          "type": tuple,
          "default": "(0,0,1,1)"
        },
        "c": {
          "type": tuple,
          "default": "(0,1,1,0)"
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        }
      }
    },
    "skimage.draw.polygon2mask": {
      "name": "polygon2mask",
      "function_name": "polygon2mask",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "mask"
        }
      },
      "function": draw.polygon2mask,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.polygon2mask",
      "parameters": {
        "image_shape": {
          "type": tuple,
          "default": None
        },
        "polygon": {
          "type": "coordinates_nd",
          "default": None
        }
      }
    },
    "skimage.draw.polygon_perimeter": {
      "name": "polygon_perimeter",
      "function_name": "polygon_perimeter",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.polygon_perimeter,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.polygon_perimeter",
      "parameters": {
        "r": {
          "type": tuple,
          "default": "(0,0,1,1)"
        },
        "c": {
          "type": tuple,
          "default": "(0,1,1,0)"
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        },
        "clip": {
          "type": bool,
          "default": False
        }
      }
    },
    "skimage.draw.random_shapes": {
      "name": "random_shapes",
      "function_name": "random_shapes",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": draw.random_shapes,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.random_shapes",
      "parameters": {
        "image_shape": {
          "type": tuple,
          "default": None
        },
        "max_shapes": {
          "type": int,
          "default": 1
        },
        "min_shapes": {
          "type": int,
          "default": 1
        },
        "min_size": {
          "type": int,
          "default": 2
        },
        "max_size": {
          "type": (int, None),
          "default": None
        },
        "num_channels": {
          "type": int,
          "default": 3
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        },
        "intensity_range": {
          "type": (tuple, None),
          "default": None
        },
        "allow_overlap": {
          "type": bool,
          "default": False
        },
        "num_trials": {
          "type": int,
          "default": 100
        },
        "rng": {
          "type": (int, None),
          "default": None
        },
        "channel_axis": {
          "type": int,
          "default": -1
        }
      }
    },
    "skimage.draw.rectangle": {
      "name": "rectangle",
      "function_name": "rectangle",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.rectangle,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.rectangle",
      "parameters": {
        "start": {
          "type": tuple,
          "default": None
        },
        "end": {
          "type": (tuple, None),
          "default": None
        },
        "extent": {
          "type": (tuple, None),
          "default": None
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        }
      }
    },
    "skimage.draw.rectangle_perimeter": {
      "name": "rectangle_perimeter",
      "function_name": "rectangle_perimeter",
      "type": "function",
      "outputs": {
        "0": {
          "type": "coordinates",
          "name": "rr, cc"
        }
      },
      "function": draw.rectangle_perimeter,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.rectangle_perimeter",
      "parameters": {
        "start": {
          "type": tuple,
          "default": None
        },
        "end": {
          "type": (tuple, None),
          "default": None
        },
        "extent": {
          "type": (tuple, None),
          "default": None
        },
        "shape": {
          "type": (tuple, None),
          "default": None
        },
        "clip": {
          "type": bool,
          "default": False
        }
      }
    },
    "skimage.draw.set_color": {
      "name": "set_color",
      "function_name": "set_color",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": draw.set_color,
      "help": "https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.set_color",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "coords": {
          "type": "coordinates",
          "default": None
        },
        "color": {
          "type": (int, float),
          "default": None
        },
        "alpha": {
          "type": int,
          "default": 1
        }
      }
    }
  },
  "exposure": {
    "skimage.exposure.adjust_gamma": {
      "name": "adjust_gamma",
      "function_name": "adjust_gamma",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": exposure.adjust_gamma,
      "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "gamma": {
          "type": float,
          "default": 1
        },
        "gain": {
          "type": float,
          "default": 1
        }
      }
    },
    "skimage.exposure.adjust_log": {
      "name": "adjust_log",
      "function_name": "adjust_log",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": exposure.adjust_log,
      "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_log",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "gain": {
          "type": float,
          "default": 1
        },
        "inv": {
          "type": bool,
          "default": False
        }
      }
    },
    "skimage.exposure.adjust_sigmoid": {
      "name": "adjust_sigmoid",
      "function_name": "adjust_sigmoid",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": exposure.adjust_sigmoid,
      "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_sigmoid",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "cutoff": {
          "type": float,
          "default": 0.5
        },
        "gain": {
          "type": float,
          "default": 10
        },
        "inv": {
          "type": bool,
          "default": False
        }
      }
    },
    # "cumulative_distribution": {
    #   "name": "cumulative_distribution",
    #   "function_name": "cumulative_distribution",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     },
    #     "1": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": exposure.cumulative_distribution,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.cumulative_distribution",
    #   "parameters": {
    #     "nbins": {
    #       "type": int,
    #       "default": 256
    #     }
    #   }
    # },
    "skimage.exposure.equalize_adapthist": {
      "name": "equalize_adapthist",
      "function_name": "equalize_adapthist",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": exposure.equalize_adapthist,
      "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "kernel_size": {
          "type": (int, tuple, None),
          "default": None
        },
        "clip_limit": {
          "type": float,
          "default": 0.01
        },
        "nbins": {
          "type": int,
          "default": 256
        }
      }
    },
    "skimage.exposure.equalize_hist": {
      "name": "equalize_hist",
      "function_name": "equalize_hist",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": exposure.equalize_hist,
      "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "nbins": {
          "type": int,
          "default": 256
        },
        "mask": {
          "type": ("image",None),
          "default": None
        }
      }
    },
    # "histogram": {
    #   "name": "histogram",
    #   "function_name": "histogram",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     },
    #     "1": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": exposure.histogram,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram",
    #   "parameters": {
    #     "nbins": {
    #       "type": int,
    #       "default": 256
    #     },
    #     "source_range": {
    #       "type": str,
    #       "default": "image"
    #     },
    #     "normalize": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "channel_axis": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "is_low_contrast": {
    #   "name": "is_low_contrast",
    #   "function_name": "is_low_contrast",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "bool_",
    #       "name": "result"
    #     }
    #   },
    #   "function": exposure.is_low_contrast,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.is_low_contrast",
    #   "parameters": {
    #     "fraction_threshold": {
    #       "type": float,
    #       "default": 0.05
    #     },
    #     "lower_percentile": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "upper_percentile": {
    #       "type": int,
    #       "default": 99
    #     },
    #     "method": {
    #       "type": str,
    #       "default": "linear"
    #     }
    #   }
    # },
    "skimage.exposure.match_histograms": {
      "name": "match_histograms",
      "function_name": "match_histograms",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": exposure.match_histograms,
      "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.match_histograms",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "reference": {
          "type": "image",
          "default": None
        },
        "channel_axis": {
          "type": (int,None),
          "default": None
        }
      }
    },
    "skimage.exposure.rescale_intensity": {
      "name": "rescale_intensity",
      "function_name": "rescale_intensity",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": exposure.rescale_intensity,
      "help": "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "in_range": {
          "type": str,
          "default":"image"
        },
        "out_range": {
          "type": str,
          "default": "dtype"
        }
      }
    }
  },
  "feature": {
    "skimage.feature.blob_dog": {
      "name": "blob_dog",
      "function_name": "blob_dog",
      "type": "function",
      "outputs": {
        "0": {
          "type": "points",
          "formats": (("Pos","Pos","Size"), ("Pos","Pos","Pos","Size")),
          "name": "points"
        }
      },
      "function": feature.blob_dog,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_dog",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "min_sigma": {
          "type": (float, tuple),
          "default": 1
        },
        "max_sigma": {
          "type": (float,tuple),
          "default": 50
        },
        "sigma_ratio": {
          "type": float,
          "default": 1.6
        },
        "threshold": {
          "type": (float, None),
          "default": 0.5
        },
        "overlap": {
          "type": float,
          "default": 0.5
        },
        "threshold_rel": {
          "type": (float, None),
          "default": None
        },
        "exclude_border": {
          "type": (int, tuple, bool),
          "default": False
        }
      }
    },
    "skimage.feature.blob_doh": {
      "name": "blob_doh",
      "function_name": "blob_doh",
      "type": "function",
      "outputs": {
        "0": {
          "type": "points",
          "formats": (("Pos","Pos","Size"), ("Pos","Pos","Pos","Size")),
          "name": "points"
        }
      },
      "function": feature.blob_doh,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_doh",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "min_sigma": {
          "type": (float, tuple),
          "default": 1
        },
        "max_sigma": {
          "type": (float, tuple),
          "default": 30
        },
        "num_sigma": {
          "type": int,
          "default": 10
        },
        "threshold": {
          "type": (float, None),
          "default": 0.01
        },
        "overlap": {
          "type": float,
          "default": 0.5
        },
        "log_scale": {
          "type": bool,
          "default": False
        },
        "threshold_rel": {
          "type": (float, None),
          "default": None
        }
      }
    },
    "skimage.feature.blob_log": {
      "name": "blob_log",
      "function_name": "blob_log",
      "type": "function",
      "outputs": {
        "0": {
          "type": "points",
          "formats": (("Pos","Pos","Size"), ("Pos","Pos","Pos","Size")),
          "name": "points"
        }
      },
      "function": feature.blob_log,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "min_sigma": {
          "type": (float, tuple),
          "default": 1
        },
        "max_sigma": {
          "type": (float, tuple),
          "default": 50
        },
        "num_sigma": {
          "type": int,
          "default": 10
        },
        "threshold": {
          "type": (float, None),
          "default": 0.2
        },
        "overlap": {
          "type": float,
          "default": 0.5
        },
        "log_scale": {
          "type": bool,
          "default": False
        },
        "threshold_rel": {
          "type": (float, None),
          "default": None
        },
        "exclude_border": {
          "type": (int, tuple, bool),
          "default": False
        }
      }
    },
    "skimage.feature.canny": {
      "name": "canny",
      "function_name": "canny",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": feature.canny,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "sigma": {
          "type": float,
          "default": 1.0
        },
        "low_threshold": {
          "type": (float, None),
          "default": None
        },
        "high_threshold": {
          "type": (float, None),
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        },
        "use_quantiles": {
          "type": bool,
          "default": False
        },
        "mode": {
          "type": str,
          "default": "constant"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.feature.corner_fast": {
      "name": "corner_fast",
      "function_name": "corner_fast",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": feature.corner_fast,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_fast",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "n": {
          "type": int,
          "default": 12
        },
        "threshold": {
          "type": float,
          "default": 0.15
        }
      }
    },
    # "corner_foerstner": {
    #   "name": "corner_foerstner",
    #   "function_name": "corner_foerstner",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     },
    #     "1": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": feature.corner_foerstner,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_foerstner",
    #   "parameters": {
    #     "sigma": {
    #       "type": int,
    #       "default": 1
    #     }
    #   }
    # },
    "skimage.feature.corner_harris": {
      "name": "corner_harris",
      "function_name": "corner_harris",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": feature.corner_harris,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_harris",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "method": {
          "type": str,
          "default": "k"
        },
        "k": {
          "type": float,
          "default": 0.05
        },
        "eps": {
          "type": float,
          "default": 1e-06
        },
        "sigma": {
          "type": float,
          "default": 1
        }
      }
    },
    "skimage.feature.corner_kitchen_rosenfeld": {
      "name": "corner_kitchen_rosenfeld",
      "function_name": "corner_kitchen_rosenfeld",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": feature.corner_kitchen_rosenfeld,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_kitchen_rosenfeld",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mode": {
          "type": str,
          "default": "constant"
        },
        "cval": {
          "type": float,
          "default": 0
        }
      }
    },
    "skimage.feature.corner_moravec": {
      "name": "corner_moravec",
      "function_name": "corner_moravec",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": feature.corner_moravec,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_moravec",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "window_size": {
          "type": int,
          "default": 1
        }
      }
    },
    # "corner_orientations": {
    #   "name": "corner_orientations",
    #   "function_name": "corner_orientations",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.corner_orientations,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_orientations",
    #   "parameters": {
    #     "corners": {
    #       "type": type,
    #       "default": None
    #     },
    #     "mask": {
    #       "type": type,
    #       "default": None
    #     }
    #   }
    # },
    "skimage.feature.corner_peaks": {
      "name": "corner_peaks",
      "function_name": "corner_peaks",
      "type": "function",
      "outputs": {
        "0": {
          "type": "points",
          "formats": (("Pos","Pos"), ("Pos","Pos","Pos")),
          "name": "image"
        }
      },
      "function": feature.corner_peaks,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_peaks",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "min_distance": {
          "type": int,
          "default": 1
        },
        "threshold_abs": {
          "type": (float, None),
          "default": None
        },
        "threshold_rel": {
          "type": (float, None),
          "default": None
        },
        "exclude_border": {
          "type": (int, tuple, bool),
          "default": True
        },
        # "indices": {
        #   "type": bool,
        #   "default": True
        # },
        "num_peaks": {
          "type": int,
          "default": 2147483647
        },
        "footprint": {
          "type": (int, None),
          "default": None
        },
        "labels": {
          "type": ("labels", None),
          "default": None
        },
        "num_peaks_per_label": {
          "type": int,
          "default": 2147483647
        },
        "p_norm": {
          "type": float,
          "default": 1
        }
      }
    },
    "skimage.feature.corner_shi_tomasi": {
      "name": "corner_shi_tomasi",
      "function_name": "corner_shi_tomasi",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": feature.corner_shi_tomasi,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_shi_tomasi",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "sigma": {
          "type": float,
          "default": 1
        }
      }
    },
    "skimage.feature.corner_subpix": {
      "name": "corner_subpix",
      "function_name": "corner_subpix",
      "type": "function",
      "outputs": None,
      "function": feature.corner_subpix,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_subpix",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "points": {
          "type": "points",
          "default": None
        },
        "window_size": {
          "type": int,
          "default": 11
        },
        "alpha": {
          "type": float,
          "default": 0.99
        }
      }
    },
    # "daisy": {
    #   "name": "daisy",
    #   "function_name": "daisy",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": feature.daisy,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.daisy",
    #   "parameters": {
    #     "step": {
    #       "type": int,
    #       "default": 4
    #     },
    #     "radius": {
    #       "type": int,
    #       "default": 15
    #     },
    #     "rings": {
    #       "type": int,
    #       "default": 3
    #     },
    #     "histograms": {
    #       "type": int,
    #       "default": 8
    #     },
    #     "orientations": {
    #       "type": int,
    #       "default": 8
    #     },
    #     "normalization": {
    #       "type": str,
    #       "default": "l1"
    #     },
    #     "sigmas": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "ring_radii": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "visualize": {
    #       "type": bool,
    #       "default": False
    #     }
    #   }
    # },
    # "draw_haar_like_feature": {
    #   "name": "draw_haar_like_feature",
    #   "function_name": "draw_haar_like_feature",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.draw_haar_like_feature,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.draw_haar_like_feature",
    #   "parameters": {
    #     "r": {
    #       "type": type,
    #       "default": None
    #     },
    #     "c": {
    #       "type": type,
    #       "default": None
    #     },
    #     "width": {
    #       "type": type,
    #       "default": None
    #     },
    #     "height": {
    #       "type": type,
    #       "default": None
    #     },
    #     "feature_coord": {
    #       "type": type,
    #       "default": None
    #     },
    #     "color_positive_block": {
    #       "type": "tuple",
    #       "default": [
    #         1.0,
    #         0.0,
    #         0.0
    #       ]
    #     },
    #     "color_negative_block": {
    #       "type": "tuple",
    #       "default": [
    #         0.0,
    #         1.0,
    #         0.0
    #       ]
    #     },
    #     "alpha": {
    #       "type": float,
    #       "default": 0.5
    #     },
    #     "max_n_features": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "rng": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "draw_multiblock_lbp": {
    #   "name": "draw_multiblock_lbp",
    #   "function_name": "draw_multiblock_lbp",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.draw_multiblock_lbp,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.draw_multiblock_lbp",
    #   "parameters": {
    #     "r": {
    #       "type": type,
    #       "default": None
    #     },
    #     "c": {
    #       "type": type,
    #       "default": None
    #     },
    #     "width": {
    #       "type": type,
    #       "default": None
    #     },
    #     "height": {
    #       "type": type,
    #       "default": None
    #     },
    #     "lbp_code": {
    #       "type": int,
    #       "default": 0
    #     },
    #     "color_greater_block": {
    #       "type": "tuple",
    #       "default": [
    #         1,
    #         1,
    #         1
    #       ]
    #     },
    #     "color_less_block": {
    #       "type": "tuple",
    #       "default": [
    #         0,
    #         0.69,
    #         0.96
    #       ]
    #     },
    #     "alpha": {
    #       "type": float,
    #       "default": 0.5
    #     }
    #   }
    # },
    # "fisher_vector": {
    #   "name": "fisher_vector",
    #   "function_name": "fisher_vector",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.fisher_vector,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.fisher_vector",
    #   "parameters": {
    #     "descriptors": {
    #       "type": type,
    #       "default": None
    #     },
    #     "gmm": {
    #       "type": type,
    #       "default": None
    #     },
    #     "improved": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "alpha": {
    #       "type": float,
    #       "default": 0.5
    #     }
    #   }
    # },
    # "graycomatrix": {
    #   "name": "graycomatrix",
    #   "function_name": "graycomatrix",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.graycomatrix,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix",
    #   "parameters": {
    #     "distances": {
    #       "type": type,
    #       "default": None
    #     },
    #     "angles": {
    #       "type": type,
    #       "default": None
    #     },
    #     "levels": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "symmetric": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "normed": {
    #       "type": bool,
    #       "default": False
    #     }
    #   }
    # },
    # "graycoprops": {
    #   "name": "graycoprops",
    #   "function_name": "graycoprops",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.graycoprops,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops",
    #   "parameters": {
    #     "P": {
    #       "type": type,
    #       "default": None
    #     },
    #     "prop": {
    #       "type": str,
    #       "default": "contrast"
    #     }
    #   }
    # },
    # "haar_like_feature": {
    #   "name": "haar_like_feature",
    #   "function_name": "haar_like_feature",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.haar_like_feature,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.haar_like_feature",
    #   "parameters": {
    #     "int_image": {
    #       "type": type,
    #       "default": None
    #     },
    #     "r": {
    #       "type": type,
    #       "default": None
    #     },
    #     "c": {
    #       "type": type,
    #       "default": None
    #     },
    #     "width": {
    #       "type": type,
    #       "default": None
    #     },
    #     "height": {
    #       "type": type,
    #       "default": None
    #     },
    #     "feature_type": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "feature_coord": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "haar_like_feature_coord": {
    #   "name": "haar_like_feature_coord",
    #   "function_name": "haar_like_feature_coord",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.haar_like_feature_coord,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.haar_like_feature_coord",
    #   "parameters": {
    #     "width": {
    #       "type": type,
    #       "default": None
    #     },
    #     "height": {
    #       "type": type,
    #       "default": None
    #     },
    #     "feature_type": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "hessian_matrix": {
    #   "name": "hessian_matrix",
    #   "function_name": "hessian_matrix",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "list",
    #       "name": "result"
    #     }
    #   },
    #   "function": feature.hessian_matrix,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hessian_matrix",
    #   "parameters": {
    #     "sigma": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "mode": {
    #       "type": str,
    #       "default": "constant"
    #     },
    #     "cval": {
    #       "type": int,
    #       "default": 0
    #     },
    #     "order": {
    #       "type": str,
    #       "default": "rc"
    #     },
    #     "use_gaussian_derivatives": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "hessian_matrix_det": {
    #   "name": "hessian_matrix_det",
    #   "function_name": "hessian_matrix_det",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": feature.hessian_matrix_det,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hessian_matrix_det",
    #   "parameters": {
    #     "sigma": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "approximate": {
    #       "type": bool,
    #       "default": True
    #     }
    #   }
    # },
    # "hessian_matrix_eigvals": {
    #   "name": "hessian_matrix_eigvals",
    #   "function_name": "hessian_matrix_eigvals",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.hessian_matrix_eigvals,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hessian_matrix_eigvals",
    #   "parameters": {
    #     "H_elems": {
    #       "type": type,
    #       "default": None
    #     }
    #   }
    # },
    # "hog": {
    #   "name": "hog",
    #   "function_name": "hog",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": feature.hog,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog",
    #   "parameters": {
    #     "orientations": {
    #       "type": int,
    #       "default": 9
    #     },
    #     "pixels_per_cell": {
    #       "type": "tuple",
    #       "default": [
    #         8,
    #         8
    #       ]
    #     },
    #     "cells_per_block": {
    #       "type": "tuple",
    #       "default": [
    #         3,
    #         3
    #       ]
    #     },
    #     "block_norm": {
    #       "type": str,
    #       "default": "L2-Hys"
    #     },
    #     "visualize": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "transform_sqrt": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "feature_vector": {
    #       "type": bool,
    #       "default": True
    #     },
    #     "channel_axis": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "learn_gmm": {
    #   "name": "learn_gmm",
    #   "function_name": "learn_gmm",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.learn_gmm,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.learn_gmm",
    #   "parameters": {
    #     "descriptors": {
    #       "type": type,
    #       "default": None
    #     },
    #     "n_modes": {
    #       "type": int,
    #       "default": 32
    #     },
    #     "gm_args": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "local_binary_pattern": {
    #   "name": "local_binary_pattern",
    #   "function_name": "local_binary_pattern",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.local_binary_pattern,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern",
    #   "parameters": {
    #     "P": {
    #       "type": type,
    #       "default": None
    #     },
    #     "R": {
    #       "type": type,
    #       "default": None
    #     },
    #     "method": {
    #       "type": str,
    #       "default": "default"
    #     }
    #   }
    # },
    # "match_descriptors": {
    #   "name": "match_descriptors",
    #   "function_name": "match_descriptors",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.match_descriptors,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.match_descriptors",
    #   "parameters": {
    #     "descriptors1": {
    #       "type": type,
    #       "default": None
    #     },
    #     "descriptors2": {
    #       "type": type,
    #       "default": None
    #     },
    #     "metric": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "p": {
    #       "type": int,
    #       "default": 2
    #     },
    #     "max_distance": {
    #       "type": float,
    #       "default": 2147483647
    #     },
    #     "cross_check": {
    #       "type": bool,
    #       "default": True
    #     },
    #     "max_ratio": {
    #       "type": float,
    #       "default": 1.0
    #     }
    #   }
    # },
    # "match_template": {
    #   "name": "match_template",
    #   "function_name": "match_template",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.match_template,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.match_template",
    #   "parameters": {
    #     "template": {
    #       "type": type,
    #       "default": None
    #     },
    #     "pad_input": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "mode": {
    #       "type": str,
    #       "default": "constant"
    #     },
    #     "constant_values": {
    #       "type": int,
    #       "default": 0
    #     }
    #   }
    # },
    # "multiblock_lbp": {
    #   "name": "multiblock_lbp",
    #   "function_name": "multiblock_lbp",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.multiblock_lbp,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.multiblock_lbp",
    #   "parameters": {
    #     "int_image": {
    #       "type": type,
    #       "default": None
    #     },
    #     "r": {
    #       "type": type,
    #       "default": None
    #     },
    #     "c": {
    #       "type": type,
    #       "default": None
    #     },
    #     "width": {
    #       "type": type,
    #       "default": None
    #     },
    #     "height": {
    #       "type": type,
    #       "default": None
    #     }
    #   }
    # },
    # "multiscale_basic_features": {
    #   "name": "multiscale_basic_features",
    #   "function_name": "multiscale_basic_features",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": feature.multiscale_basic_features,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.multiscale_basic_features",
    #   "parameters": {
    #     "intensity": {
    #       "type": bool,
    #       "default": True
    #     },
    #     "edges": {
    #       "type": bool,
    #       "default": True
    #     },
    #     "texture": {
    #       "type": bool,
    #       "default": True
    #     },
    #     "sigma_min": {
    #       "type": float,
    #       "default": 0.5
    #     },
    #     "sigma_max": {
    #       "type": int,
    #       "default": 16
    #     },
    #     "num_sigma": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "num_workers": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "channel_axis": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    "skimage.feature.peak_local_max": {
      "name": "peak_local_max",
      "function_name": "peak_local_max",
      "type": "function",
      "outputs": {
        "0": {
          "type": "points",
          "formats": (("Pos","Pos"), ("Pos","Pos","Pos")),
          "name": "points"
        }
      },
      "function": feature.peak_local_max,
      "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "min_distance": {
          "type": int,
          "default": 1
        },
        "threshold_abs": {
          "type": (float, None),
          "default": None
        },
        "threshold_rel": {
          "type": (float, None),
          "default": None
        },
        "exclude_border": {
          "type": (int, tuple, bool),
          "default": True
        },
        "num_peaks": {
          "type": int,
          "default": 2147483647
        },
        "footprint": {
          "type": (int, None),
          "default": None
        },
        "labels": {
          "type": ("labels", None),
          "default": None
        },
        "num_peaks_per_label": {
          "type": int,
          "default": 2147483647
        },
        "p_norm": {
          "type": float,
          "default": 1
        }
      }
    },
    # "plot_matched_features": {
    #   "name": "plot_matched_features",
    #   "function_name": "plot_matched_features",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.plot_matched_features,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.plot_matched_features",
    #   "parameters": {
    #     "image0": {
    #       "type": type,
    #       "default": None
    #     },
    #     "image1": {
    #       "type": type,
    #       "default": None
    #     },
    #     "keypoints0": {
    #       "type": type,
    #       "default": None
    #     },
    #     "keypoints1": {
    #       "type": type,
    #       "default": None
    #     },
    #     "matches": {
    #       "type": type,
    #       "default": None
    #     },
    #     "ax": {
    #       "type": type,
    #       "default": None
    #     },
    #     "keypoints_color": {
    #       "type": str,
    #       "default": "k"
    #     },
    #     "matches_color": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "only_matches": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "alignment": {
    #       "type": str,
    #       "default": "horizontal"
    #     }
    #   }
    # },
    # "shape_index": {
    #   "name": "shape_index",
    #   "function_name": "shape_index",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": feature.shape_index,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.shape_index",
    #   "parameters": {
    #     "sigma": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "mode": {
    #       "type": str,
    #       "default": "constant"
    #     },
    #     "cval": {
    #       "type": int,
    #       "default": 0
    #     }
    #   }
    # },
    # "structure_tensor": {
    #   "name": "structure_tensor",
    #   "function_name": "structure_tensor",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "list",
    #       "name": "result"
    #     }
    #   },
    #   "function": feature.structure_tensor,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.structure_tensor",
    #   "parameters": {
    #     "sigma": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "mode": {
    #       "type": str,
    #       "default": "constant"
    #     },
    #     "cval": {
    #       "type": int,
    #       "default": 0
    #     },
    #     "order": {
    #       "type": str,
    #       "default": "rc"
    #     }
    #   }
    # },
    # "structure_tensor_eigenvalues": {
    #   "name": "structure_tensor_eigenvalues",
    #   "function_name": "structure_tensor_eigenvalues",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": feature.structure_tensor_eigenvalues,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.structure_tensor_eigenvalues",
    #   "parameters": {
    #     "A_elems": {
    #       "type": type,
    #       "default": None
    #     }
    #   }
    # }
  },
  "filters": {
    "skimage.filters.butterworth": {
      "name": "butterworth",
      "function_name": "butterworth",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.butterworth,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.butterworth",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "cutoff_frequency_ratio": {
          "type": float,
          "default": 0.005
        },
        "high_pass": {
          "type": bool,
          "default": True
        },
        "order": {
          "type": float,
          "default": 2.0
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        },
        "squared_butterworth": {
          "type": bool,
          "default": True
        },
        "npad": {
          "type": int,
          "default": 0
        }
      }
    },
    "skimage.filters.farid": {
      "name": "farid",
      "function_name": "farid",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.farid,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        },
        "axis": {
          "type": (int, tuple, None),
          "default": None
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.filters.farid_h": {
      "name": "farid_h",
      "function_name": "farid_h",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.farid_h,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_h",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.farid_v": {
      "name": "farid_v",
      "function_name": "farid_v",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.farid_v,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_v",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.frangi": {
      "name": "frangi",
      "function_name": "frangi",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.frangi,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.frangi",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "sigmas": {
          "type": range,
          "default": "range(1, 10, 2)"
        },
        "scale_range": {
          "type": (tuple, None),
          "default": None
        },
        "scale_step": {
          "type": (float, None),
          "default": None
        },
        "alpha": {
          "type": float,
          "default": 0.5
        },
        "beta": {
          "type": float,
          "default": 0.5
        },
        "gamma": {
          "type": (tuple, None),
          "default": None
        },
        "black_ridges": {
          "type": bool,
          "default": True
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": int,
          "default": 0
        }
      }
    },
    "skimage.filters.gaussian": {
      "name": "gaussian",
      "function_name": "gaussian",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.gaussian,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.gaussian",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "sigma": {
          "type": float,
          "default": 1.0
        },
        "mode": {
          "type": str,
          "default": "nearest"
        },
        "cval": {
          "type": int,
          "default": 0
        },
        "preserve_range": {
          "type": bool,
          "default": False
        },
        "truncate": {
          "type": float,
          "default": 4.0
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        },
        # "out": {
        #   "type": None,
        #   "default": None
        # }
      }
    },
    "skimage.filters.hessian": {
      "name": "hessian",
      "function_name": "hessian",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.hessian,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.hessian",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "sigmas": {
          "type": range,
          "default": "range(1, 10, 2)"
        },
        "scale_range": {
          "type": (tuple, None),
          "default": None
        },
        "scale_step": {
          "type": (float, None),
          "default": None
        },
        "alpha": {
          "type": float,
          "default": 0.5
        },
        "beta": {
          "type": float,
          "default": 0.5
        },
        "gamma": {
          "type": int,
          "default": 15
        },
        "black_ridges": {
          "type": bool,
          "default": True
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": int,
          "default": 0
        }
      }
    },
    "skimage.filters.laplace": {
      "name": "laplace",
      "function_name": "laplace",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.laplace,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.laplace",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "ksize": {
          "type": int,
          "default": 3
        },
        "mask": {
          "type": ("mask",None),
          "default": None
        }
      }
    },
    "skimage.filters.median": {
      "name": "median",
      "function_name": "median",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.median,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.median",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": None,
        #   "default": None
        # },
        "mode": {
          "type": str,
          "default": "nearest"
        },
        "cval": {
          "type": float,
          "default": 0.0
        },
        "behavior": {
          "type": str,
          "default": "ndimage"
        }
      }
    },
    "skimage.filters.meijering": {
      "name": "meijering",
      "function_name": "meijering",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.meijering,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.meijering",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "sigmas": {
          "type": range,
          "default": "range(1, 10, 2)"
        },
        "alpha": {
          "type": (float, None),
          "default": None
        },
        "black_ridges": {
          "type": bool,
          "default": True
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": int,
          "default": 0
        }
      }
    },
    "skimage.filters.prewitt": {
      "name": "prewitt",
      "function_name": "prewitt",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.prewitt,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        },
        "axis": {
          "type": (int, tuple, None),
          "default": None
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.filters.prewitt_h": {
      "name": "prewitt_h",
      "function_name": "prewitt_h",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.prewitt_h,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_h",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.prewitt_v": {
      "name": "prewitt_v",
      "function_name": "prewitt_v",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.prewitt_v,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_v",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.rank_order": {
      "name": "rank_order",
      "function_name": "rank_order",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        },
        "1": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.rank_order,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.rank_order",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        }          
      }
    },
    "skimage.filters.roberts": {
      "name": "roberts",
      "function_name": "roberts",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.roberts,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.roberts_neg_diag": {
      "name": "roberts_neg_diag",
      "function_name": "roberts_neg_diag",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.roberts_neg_diag,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts_neg_diag",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.roberts_pos_diag": {
      "name": "roberts_pos_diag",
      "function_name": "roberts_pos_diag",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.roberts_pos_diag,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts_pos_diag",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.sato": {
      "name": "sato",
      "function_name": "sato",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.sato,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sato",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "sigmas": {
          "type": range,
          "default": "range(1, 10, 2)"
        },
        "black_ridges": {
          "type": bool,
          "default": True
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": int,
          "default": 0
        }
      }
    },
    "skimage.filters.scharr": {
      "name": "scharr",
      "function_name": "scharr",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.scharr,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        },
        "axis": {
          "type": (int, tuple, None),
          "default": None
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.filters.scharr_h": {
      "name": "scharr_h",
      "function_name": "scharr_h",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.scharr_h,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_h",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.scharr_v": {
      "name": "scharr_v",
      "function_name": "scharr_v",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.scharr_v,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_v",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.sobel": {
      "name": "sobel",
      "function_name": "sobel",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.sobel,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        },
        "axis": {
          "type": (int, tuple, None),
          "default": None
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.filters.sobel_h": {
      "name": "sobel_h",
      "function_name": "sobel_h",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.sobel_h,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_h",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.sobel_v": {
      "name": "sobel_v",
      "function_name": "sobel_v",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.sobel_v,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_v",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.filters.threshold_isodata": {
      "name": "threshold_isodata",
      "function_name": "threshold_isodata",
      "type": "function",
      "outputs": {
        "0": {
          "type": "scalar",
          "name": "value"
        }
      },
      "function": filters.threshold_isodata,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_isodata",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "nbins": {
          "type": int,
          "default": 256
        },
        "return_all": {
          "type": bool,
          "default": False
        },
        # "hist": {
        #   "type": None,
        #   "default": None
        # }
      }
    },
    "skimage.filters.threshold_li": {
      "name": "threshold_li",
      "function_name": "threshold_li",
      "type": "function",
      "outputs": {
        "0": {
          "type": "scalar",
          "name": "value"
        }
      },
      "function": filters.threshold_li,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_li",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "tolerance": {
          "type": (float, None),
          "default": None
        },
        "initial_guess": {
          "type": ("function", None),
          "default": None
        },
        "iter_callback": {
          "type": ("function", None),
          "default": None
        }
      }
    },
    "skimage.filters.threshold_local": {
      "name": "threshold_local",
      "function_name": "threshold_local",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.threshold_local,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_local",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "block_size": {
          "type": int,
          "default": 3
        },
        "method": {
          "type": str,
          "default": "gaussian"
        },
        "offset": {
          "type": int,
          "default": 0
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "param": {
          "type": (int, "function", None),
          "default": None
        },
        "cval": {
          "type": int,
          "default": 0
        }
      }
    },
    "skimage.filters.threshold_mean": {
      "name": "threshold_mean",
      "function_name": "threshold_mean",
      "type": "function",
      "outputs": {
        "0": {
          "type": "scalar",
          "name": "value"
        }
      },
      "function": filters.threshold_mean,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_mean",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        }          
      }
    },
    "skimage.filters.threshold_minimum": {
      "name": "threshold_minimum",
      "function_name": "threshold_minimum",
      "type": "function",
      "outputs": {
        "0": {
          "type": "scalar",
          "name": "value"
        }
      },
      "function": filters.threshold_minimum,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_minimum",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "nbins": {
          "type": int,
          "default": 256
        },
        "max_num_iter": {
          "type": int,
          "default": 10000
        },
        # "hist": {
        #   "type": None,
        #   "default": None
        # }
      }
    },
    "skimage.filters.threshold_multiotsu": {
      "name": "threshold_multiotsu",
      "function_name": "threshold_multiotsu",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.threshold_multiotsu,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_multiotsu",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "classes": {
          "type": int,
          "default": 3
        },
        "nbins": {
          "type": int,
          "default": 256
        },
        # "hist": {
        #   "type": None,
        #   "default": None
        # }
      }
    },
    "skimage.filters.threshold_niblack": {
      "name": "threshold_niblack",
      "function_name": "threshold_niblack",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.threshold_niblack,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_niblack",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "window_size": {
          "type": int,
          "default": 15
        },
        "k": {
          "type": float,
          "default": 0.2
        }
      }
    },
    "skimage.filters.threshold_otsu": {
      "name": "threshold_otsu",
      "function_name": "threshold_otsu",
      "type": "function",
      "outputs": {
        "0": {
          "type": "scalar",
          "name": "value"
        }
      },
      "function": filters.threshold_otsu,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "nbins": {
          "type": int,
          "default": 256
        },
        # "hist": {
        #   "type": None,
        #   "default": None
        # }
      }
    },
    "skimage.filters.threshold_sauvola": {
      "name": "threshold_sauvola",
      "function_name": "threshold_sauvola",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.threshold_sauvola,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "window_size": {
          "type": int,
          "default": 15
        },
        "k": {
          "type": float,
          "default": 0.2
        },
        "r": {
          "type": (float, None),
          "default": None
        }
      }
    },
    "skimage.filters.threshold_triangle": {
      "name": "threshold_triangle",
      "function_name": "threshold_triangle",
      "type": "function",
      "outputs": {
        "0": {
          "type": "scalar",
          "name": "value"
        }
      },
      "function": filters.threshold_triangle,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_triangle",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "nbins": {
          "type": int,
          "default": 256
        }
      }
    },
    "skimage.filters.threshold_yen": {
      "name": "threshold_yen",
      "function_name": "threshold_yen",
      "type": "function",
      "outputs": {
        "0": {
          "type": "scalar",
          "name": "value"
        }
      },
      "function": filters.threshold_yen,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_yen",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "nbins": {
          "type": int,
          "default": 256
        },
        # "hist": {
        #   "type": None,
        #   "default": None
        # }
      }
    },
    "skimage.filters.try_all_threshold": {
      "name": "try_all_threshold",
      "function_name": "try_all_threshold",
      "type": "function",
      "outputs": {
        "0": {
          "type": "Figure",
          "name": "result"
        },
        "1": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.try_all_threshold,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.try_all_threshold",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "figsize": {
          "type": tuple,
          "default": [
            8,
            5
          ]
        },
        "verbose": {
          "type": bool,
          "default": True
        }
      }
    },
    "skimage.filters.unsharp_mask": {
      "name": "unsharp_mask",
      "function_name": "unsharp_mask",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": filters.unsharp_mask,
      "help": "https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.unsharp_mask",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "radius": {
          "type": float,
          "default": 1.0
        },
        "amount": {
          "type": float,
          "default": 1.0
        },
        "preserve_range": {
          "type": bool,
          "default": False
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    }
  },
  "morphology": {
    "skimage.morphology.area_closing": {
      "name": "area_closing",
      "function_name": "area_closing",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.area_closing,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.area_closing",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "area_threshold": {
          "type": int,
          "default": 64
        },
        "connectivity": {
          "type": int,
          "default": 1
        },
        # "parent": {
        #   "type": ("image", None),
        #   "default": None
        # },
        # "tree_traverser": {
        #   "type": (int, None),
        #   "default": None
        # }
      }
    },
    "skimage.morphology.area_opening": {
      "name": "area_opening",
      "function_name": "area_opening",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.area_opening,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.area_opening",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "area_threshold": {
          "type": int,
          "default": 64
        },
        "connectivity": {
          "type": int,
          "default": 1
        },
        # "parent": {
        #   "type": ("image", None),
        #   "default": None
        # },
        # "tree_traverser": {
        #   "type": (int, None),
        #   "default": None
        # }
      }
    },
    "skimage.morphology.ball": {
      "name": "ball",
      "function_name": "ball",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.ball,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.ball",
      "parameters": {
        "radius": {
          "type": int,
          "default": 1
        },
        "dtype": {
          "type": type,
          "default": "uint8"
        },
        "strict_radius": {
          "type": bool,
          "default": True
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.binary_closing": {
      "name": "binary_closing",
      "function_name": "binary_closing",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.binary_closing,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_closing",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "mode": {
          "type": (str, bool),
          "default": "ignore"
        }
      }
    },
    "skimage.morphology.binary_dilation": {
      "name": "binary_dilation",
      "function_name": "binary_dilation",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.binary_dilation,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_dilation",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "mode": {
          "type": (str, bool),
          "default": "ignore"
        }
      }
    },
    "skimage.morphology.binary_erosion": {
      "name": "binary_erosion",
      "function_name": "binary_erosion",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.binary_erosion,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_erosion",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "mode": {
          "type": (str, bool),
          "default": "ignore"
        }
      }
    },
    "skimage.morphology.binary_opening": {
      "name": "binary_opening",
      "function_name": "binary_opening",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.binary_opening,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_opening",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "mode": {
          "type": (str, bool),
          "default": "ignore"
        }
      }
    },
    "skimage.morphology.black_tophat": {
      "name": "black_tophat",
      "function_name": "black_tophat",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.black_tophat,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.black_tophat",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.morphology.closing": {
      "name": "closing",
      "function_name": "closing",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.closing,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.closing",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.morphology.convex_hull_image": {
      "name": "convex_hull_image",
      "function_name": "convex_hull_image",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.convex_hull_image,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.convex_hull_image",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "offset_coordinates": {
          "type": bool,
          "default": True
        },
        "tolerance": {
          "type": float,
          "default": 1e-10
        },
        "include_borders": {
          "type": bool,
          "default": True
        }
      }
    },
    "skimage.morphology.convex_hull_object": {
      "name": "convex_hull_object",
      "function_name": "convex_hull_object",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.convex_hull_object,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.convex_hull_object",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "connectivity": {
          "type": int,
          "default": 2
        }
      }
    },
    "skimage.morphology.cube": {
      "name": "cube",
      "function_name": "cube",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.cube,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.cube",
      "parameters": {
        "width": {
          "type": int,
          "default": 1
        },
        "dtype": {
          "type": type,
          "default": "uint8"
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.diameter_closing": {
      "name": "diameter_closing",
      "function_name": "diameter_closing",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.diameter_closing,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.diameter_closing",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "diameter_threshold": {
          "type": int,
          "default": 8
        },
        "connectivity": {
          "type": int,
          "default": 1
        },
        "parent": {
          "type": ("image", int, None),
          "default": None
        },
        "tree_traverser": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.morphology.diameter_opening": {
      "name": "diameter_opening",
      "function_name": "diameter_opening",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.diameter_opening,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.diameter_opening",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "diameter_threshold": {
          "type": int,
          "default": 8
        },
        "connectivity": {
          "type": int,
          "default": 1
        },
        "parent": {
          "type": ("image", int, None),
          "default": None
        },
        "tree_traverser": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.morphology.diamond": {
      "name": "diamond",
      "function_name": "diamond",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.diamond,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.diamond",
      "parameters": {
        "radius": {
          "type": int,
          "default": 1
        },
        "dtype": {
          "type": type,
          "default": "uint8"
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.dilation": {
      "name": "dilation",
      "function_name": "dilation",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.dilation,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        # "shift_x": {
        #   "type": "PatchClassRepr",
        #   "default": "<DEPRECATED>"
        # },
        # "shift_y": {
        #   "type": "PatchClassRepr",
        #   "default": "<DEPRECATED>"
        # },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.morphology.disk": {
      "name": "disk",
      "function_name": "disk",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.disk,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.disk",
      "parameters": {
        "radius": {
          "type": int,
          "default": 1
        },
        "dtype": {
          "type": type,
          "default": "uint8"
        },
        "strict_radius": {
          "type": bool,
          "default": True
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.ellipse": {
      "name": "ellipse",
      "function_name": "ellipse",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.ellipse,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.ellipse",
      "parameters": {
        "width": {
          "type": int,
          "name": "width"
        },
        "height": {
          "type": int,
          "name": "heigh"
        },
        "dtype": {
          "type": type,
          "default": "uint8"
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.erosion": {
      "name": "erosion",
      "function_name": "erosion",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.erosion,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        # "shift_x": {
        #   "type": "PatchClassRepr",
        #   "default": "<DEPRECATED>"
        # },
        # "shift_y": {
        #   "type": "PatchClassRepr",
        #   "default": "<DEPRECATED>"
        # },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.morphology.flood": {
      "name": "flood",
      "function_name": "flood",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.flood,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.flood",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "seed_point": {
          "type": tuple,
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        "connectivity": {
          "type": (int, None),
          "default": None
        },
        "tolerance": {
          "type": (float, int, None),
          "default": None
        }
      }
    },
    "skimage.morphology.flood_fill": {
      "name": "flood_fill",
      "function_name": "flood_fill",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.flood_fill,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.flood_fill",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "seed_point": {
          "type": tuple,
          "default": None
        },
        "new_value": {
          "type": type,
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        "connectivity": {
          "type": (int, None),
          "default": None
        },
        "tolerance": {
          "type": (float, int, None),
          "default": None
        },
        # "in_place": {
        #   "type": bool,
        #   "default": False
        # }
      }
    },
    # "footprint_from_sequence": {
    #   "name": "footprint_from_sequence",
    #   "function_name": "footprint_from_sequence",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": None,
    #   "function": morphology.footprint_from_sequence,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.footprint_from_sequence",
    #   "parameters": {
    #     "footprints": {
    #       "type": type,
    #       "default": None
    #     }
    #   }
    # },
    "skimage.morphology.footprint_rectangle": {
      "name": "footprint_rectangle",
      "function_name": "footprint_rectangle",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.footprint_rectangle,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.footprint_rectangle",
      "parameters": {
        "shape": {
          "type": tuple,
          "default": "(1,1)"
        },
        "dtype": {
          "type": type,
          "default": "uint8"
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.h_maxima": {
      "name": "h_maxima",
      "function_name": "h_maxima",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.h_maxima,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.h_maxima",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "h": {
          "type": int,
          "default": 1
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.morphology.h_minima": {
      "name": "h_minima",
      "function_name": "h_minima",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.h_minima,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.h_minima",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "h": {
          "type": type,
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        }
      }
    },
    "skimage.morphology.isotropic_closing": {
      "name": "isotropic_closing",
      "function_name": "isotropic_closing",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.isotropic_closing,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.isotropic_closing",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "radius": {
          "type": float,
          "default": 1.0
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "spacing": {
          "type": (float, tuple, None),
          "default": None
        }
      }
    },
    "skimage.morphology.isotropic_dilation": {
      "name": "isotropic_dilation",
      "function_name": "isotropic_dilation",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.isotropic_dilation,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.isotropic_dilation",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "radius": {
          "type": float,
          "default": 1.0
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "spacing": {
          "type": (float, tuple, None),
          "default": None
        }
      }
    },
    "skimage.morphology.isotropic_erosion": {
      "name": "isotropic_erosion",
      "function_name": "isotropic_erosion",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.isotropic_erosion,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.isotropic_erosion",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "radius": {
          "type": type,
          "default": None
        },        
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "spacing": {
          "type": (float, tuple, None),
          "default": None
        }
      }
    },
    "skimage.morphology.isotropic_opening": {
      "name": "isotropic_opening",
      "function_name": "isotropic_opening",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.isotropic_opening,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.isotropic_opening",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "radius": {
          "type": type,
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "spacing": {
          "type": (float, tuple, None),
          "default": None
        }
      }
    },
    "skimage.morphology.label": {
      "name": "label",
      "function_name": "label",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.label,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.label",
      "parameters": {
        "label_image": {
          "type": "image",
          "default": None
        },
        "background": {
          "type": (int, None),
          "default": None
        },
        # "return_num": {
        #   "type": bool,
        #   "default": False
        # },
        "connectivity": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.morphology.local_maxima": {
      "name": "local_maxima",
      "function_name": "local_maxima",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.local_maxima,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.local_maxima",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        "connectivity": {
          "type": (int, None),
          "default": None
        },
        "indices": {
          "type": bool,
          "default": False
        },
        "allow_borders": {
          "type": bool,
          "default": True
        }
      }
    },
    "skimage.morphology.local_minima": {
      "name": "local_minima",
      "function_name": "local_minima",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.local_minima,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.local_minima",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        "connectivity": {
          "type": (int, None),
          "default": None
        },
        "indices": {
          "type": bool,
          "default": False
        },
        "allow_borders": {
          "type": bool,
          "default": True
        }
      }
    },
    # "max_tree": {
    #   "name": "max_tree",
    #   "function_name": "max_tree",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     },
    #     "1": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": morphology.max_tree,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.max_tree",
    #   "parameters": {
    #     "connectivity": {
    #       "type": int,
    #       "default": 1
    #     }
    #   }
    # },
    "skimage.morphology.max_tree_local_maxima": {
      "name": "max_tree_local_maxima",
      "function_name": "max_tree_local_maxima",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.max_tree_local_maxima,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.max_tree_local_maxima",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "connectivity": {
          "type": int,
          "default": 1
        },
        # "parent": {
        #   "type": ("image", None),
        #   "default": None
        # },
        # "tree_traverser": {
        #   "type": (int, None),
        #   "default": None
        # }
      }
    },
    # "medial_axis": {
    #   "name": "medial_axis",
    #   "function_name": "medial_axis",
    #   "type": "function",
    #   "inputs": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     }
    #   },
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": morphology.medial_axis,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.medial_axis",
    #   "parameters": {
    #     "mask": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "return_distance": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "rng": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    "skimage.morphology.mirror_footprint": {
      "name": "mirror_footprint",
      "function_name": "mirror_footprint",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.mirror_footprint,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.mirror_footprint",
      "parameters": {
        "footprint": {
          "type": "image",
          "default": None
        }
      }
    },
    "skimage.morphology.octagon": {
      "name": "octagon",
      "function_name": "octagon",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.octagon,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.octagon",
      "parameters": {
        "m": {
          "type": int,
          "default": 1
        },
        "n": {
          "type": int,
          "default": 1
        },
        "dtype": {
          "type": type,
          "default": "uint8"
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.octahedron": {
      "name": "octahedron",
      "function_name": "octahedron",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.octahedron,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.octahedron",
      "parameters": {
        "radius": {
          "type": int,
          "default": 1
        },
        "dtype": {
          "type": type,
          "default": "uint8"
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.opening": {
      "name": "opening",
      "function_name": "opening",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.opening,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.opening",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    },
    "skimage.morphology.pad_footprint": {
      "name": "pad_footprint",
      "function_name": "pad_footprint",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.pad_footprint,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.pad_footprint",
      "parameters": {
        "footprint": {
          "type": "image",
          "default": None
        },
        "pad_end": {
          "type": bool,
          "default": True
        }
      }
    },
    "skimage.morphology.reconstruction": {
      "name": "reconstruction",
      "function_name": "reconstruction",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.reconstruction,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.reconstruction",
      "parameters": {
        "seed": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": "image",
          "default": None
        },
        "method": {
          "type": str,
          "default": "dilation"
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        "offset": {
          "type": (tuple, None),
          "default": None
        }
      }
    },
    "skimage.morphology.rectangle": {
      "name": "rectangle",
      "function_name": "rectangle",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.rectangle,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.rectangle",
      "parameters": {
        "nrows": {
          "type": int,
          "default": 1
        },
        "ncols": {
          "type": int,
          "default": 1
        },
        "dtype": {
          "type": type,
          "default": "uint8"
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.remove_objects_by_distance": {
      "name": "remove_objects_by_distance",
      "function_name": "remove_objects_by_distance",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.remove_objects_by_distance,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_objects_by_distance",
      "parameters": {
        "label_image": {
          "type": "image",
          "default": None
        },
        "min_distance": {
          "type": (int, float),
          "default": None
        },
        # "priority": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "p_norm": {
          "type": (int, float),
          "default": 2
        },
        "spacing": {
          "type": "NoneType",
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.remove_small_holes": {
      "name": "remove_small_holes",
      "function_name": "remove_small_holes",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.remove_small_holes,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_holes",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "area_threshold": {
          "type": int,
          "default": 64
        },
        "connectivity": {
          "type": int,
          "default": 1
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.remove_small_objects": {
      "name": "remove_small_objects",
      "function_name": "remove_small_objects",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.remove_small_objects,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_objects",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "min_size": {
          "type": int,
          "default": 64
        },
        "connectivity": {
          "type": int,
          "default": 1
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.skeletonize": {
      "name": "skeletonize",
      "function_name": "skeletonize",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.skeletonize,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "method": {
          "type": (str, None),
          "default": None
        }
      }
    },
    "skimage.morphology.square": {
      "name": "square",
      "function_name": "square",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.square,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.square",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "width": {
          "type": type,
          "default": None
        },
        "dtype": {
          "type": type,
          "default": "<class 'numpy.uint8'>"
        },
        # "decomposition": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.morphology.star": {
      "name": "star",
      "function_name": "star",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.star,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.star",
      "parameters": {
        "a": {
          "type": int,
          "default": 1
        },
        "dtype": {
          "type": type,
          "default": "<class 'numpy.uint8'>"
        }
      }
    },
    "skimage.morphology.thin": {
      "name": "thin",
      "function_name": "thin",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.thin,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.thin",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "max_num_iter": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.morphology.white_tophat": {
      "name": "white_tophat",
      "function_name": "white_tophat",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": morphology.white_tophat,
      "help": "https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.white_tophat",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "footprint": {
          "type": ("image", None),
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": float,
          "default": 0.0
        }
      }
    }
  },
  "restoration": {
    "skimage.restoration.ball_kernel": {
      "name": "ball_kernel",
      "function_name": "ball_kernel",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.ball_kernel,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.ball_kernel",
      "parameters": {
        "radius": {
          "type": int,
          "default": 1
        },
        "ndim": {
          "type": int,
          "default": 1
        }
      }
    },
    # "calibrate_denoiser": {
    #   "name": "calibrate_denoiser",
    #   "function_name": "calibrate_denoiser",
    #   "type": "function",
    #   "outputs": None,
    #   "function": restoration.calibrate_denoiser,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.calibrate_denoiser",
    #   "parameters": {
    #     "image": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "denoise_function": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "denoise_parameters": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "stride": {
    #       "type": int,
    #       "default": 4
    #     },
    #     "approximate_loss": {
    #       "type": bool,
    #       "default": True
    #     },
    #     "extra_output": {
    #       "type": bool,
    #       "default": False
    #     }
    #   }
    # },
    # "cycle_spin": {
    #   "name": "cycle_spin",
    #   "function_name": "cycle_spin",
    #   "type": "function",
    #   "outputs": None,
    #   "function": restoration.cycle_spin,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.cycle_spin",
    #   "parameters": {
    #     "x": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "func": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "max_shifts": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "shift_steps": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "num_workers": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "func_kw": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "channel_axis": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    "skimage.restoration.denoise_bilateral": {
      "name": "denoise_bilateral",
      "function_name": "denoise_bilateral",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.denoise_bilateral,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_bilateral",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "win_size": {
          "type": (int, None),
          "default": None
        },
        "sigma_color": {
          "type": (float, None),
          "default": None
        },
        "sigma_spatial": {
          "type": float,
          "default": 1
        },
        "bins": {
          "type": int,
          "default": 10000
        },
        "mode": {
          "type": str,
          "default": "constant"
        },
        "cval": {
          "type": float,
          "default": 0
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    # "denoise_invariant": {
    #   "name": "denoise_invariant",
    #   "function_name": "denoise_invariant",
    #   "type": "function",
    #   "outputs": None,
    #   "function": restoration.denoise_invariant,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_invariant",
    #   "parameters": {
    #     "image": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "denoise_function": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "stride": {
    #       "type": int,
    #       "default": 4
    #     },
    #     "masks": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "denoiser_kwargs": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    "skimage.restoration.denoise_nl_means": {
      "name": "denoise_nl_means",
      "function_name": "denoise_nl_means",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.denoise_nl_means,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_nl_means",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "patch_size": {
          "type": int,
          "default": 7
        },
        "patch_distance": {
          "type": int,
          "default": 11
        },
        "h": {
          "type": float,
          "default": 0.1
        },
        "fast_mode": {
          "type": bool,
          "default": True
        },
        "sigma": {
          "type": float,
          "default": 0.0
        },
        "preserve_range": {
          "type": bool,
          "default": False
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.restoration.denoise_tv_bregman": {
      "name": "denoise_tv_bregman",
      "function_name": "denoise_tv_bregman",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.denoise_tv_bregman,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_bregman",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "weight": {
          "type": float,
          "default": 5.0
        },
        "max_num_iter": {
          "type": int,
          "default": 100
        },
        "eps": {
          "type": float,
          "default": 0.001
        },
        "isotropic": {
          "type": bool,
          "default": True
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.restoration.denoise_tv_chambolle": {
      "name": "denoise_tv_chambolle",
      "function_name": "denoise_tv_chambolle",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.denoise_tv_chambolle,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "weight": {
          "type": float,
          "default": 0.1
        },
        "eps": {
          "type": float,
          "default": 0.0002
        },
        "max_num_iter": {
          "type": int,
          "default": 200
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.restoration.denoise_wavelet": {
      "name": "denoise_wavelet",
      "function_name": "denoise_wavelet",
      "type": "function",
      "outputs": None,
      "function": restoration.denoise_wavelet,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_wavelet",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "sigma": {
          "type": "NoneType",
          "default": None
        },
        "wavelet": {
          "type": str,
          "default": "db1"
        },
        "mode": {
          "type": str,
          "default": "soft"
        },
        "wavelet_levels": {
          "type": (int, None),
          "default": None
        },
        "convert2ycbcr": {
          "type": bool,
          "default": False
        },
        "method": {
          "type": str,
          "default": "BayesShrink"
        },
        "rescale_sigma": {
          "type": bool,
          "default": True
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.restoration.ellipsoid_kernel": {
      "name": "ellipsoid_kernel",
      "function_name": "ellipsoid_kernel",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.ellipsoid_kernel,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.ellipsoid_kernel",
      "parameters": {
        "shape": {
          "type": "image",
          "default": None
        },
        "intensity": {
          "type": int,
          "default": 1
        }
      }
    },
    "skimage.restoration.estimate_sigma": {
      "name": "estimate_sigma",
      "function_name": "estimate_sigma",
      "type": "function",
      "outputs": {
        "0": {
          "type": "scalar",
          "name": "value"
        }
      },
      "function": restoration.estimate_sigma,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.estimate_sigma",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "average_sigmas": {
          "type": bool,
          "default": False
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.restoration.inpaint_biharmonic": {
      "name": "inpaint_biharmonic",
      "function_name": "inpaint_biharmonic",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.inpaint_biharmonic,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.inpaint_biharmonic",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mask": {
          "type": "image",
          "default": None
        },
        "split_into_regions": {
          "type": bool,
          "default": False
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.restoration.richardson_lucy": {
      "name": "richardson_lucy",
      "function_name": "richardson_lucy",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.richardson_lucy,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.richardson_lucy",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "psf": {
          "type": "image",
          "default": None
        },
        "num_iter": {
          "type": int,
          "default": 50
        },
        "clip": {
          "type": bool,
          "default": True
        },
        "filter_epsilon": {
          "type": (float, None),
          "default": None
        }
      }
    },
    "skimage.restoration.rolling_ball": {
      "name": "rolling_ball",
      "function_name": "rolling_ball",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.rolling_ball,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.rolling_ball",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "radius": {
          "type": int,
          "default": 100
        },
        "kernel": {
          "type": ("image", None),
          "default": None
        },
        "nansafe": {
          "type": bool,
          "default": False
        },
        "num_threads": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.restoration.unsupervised_wiener": {
      "name": "unsupervised_wiener",
      "function_name": "unsupervised_wiener",
      "type": "function",
      "outputs": None,
      "function": restoration.unsupervised_wiener,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.unsupervised_wiener",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "psf": {
          "type": "image",
          "default": None
        },
        "reg": {
          "type": "NoneType",
          "default": None
        },
        "user_params": {
          "type": (dict, None),
          "default": None
        },
        "is_real": {
          "type": bool,
          "default": True
        },
        "clip": {
          "type": bool,
          "default": True
        },
        "rng": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.restoration.unwrap_phase": {
      "name": "unwrap_phase",
      "function_name": "unwrap_phase",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": restoration.unwrap_phase,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.unwrap_phase",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "wrap_around": {
          "type": bool,
          "default": False
        },
        "rng": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.restoration.wiener": {
      "name": "wiener",
      "function_name": "wiener",
      "type": "function",
      "outputs": None,
      "function": restoration.wiener,
      "help": "https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.wiener",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "psf": {
          "type": "image",
          "default": None
        },
        "balance": {
          "type": float,
          "default": 1.0
        },
        "reg": {
          "type": "image",
          "default": None
        },
        "is_real": {
          "type": bool,
          "default": True
        },
        "clip": {
          "type": bool,
          "default": True
        }
      }
    }
  },
  "segmentation": {
    "skimage.segmentation.active_contour": {
      "name": "active_contour",
      "function_name": "active_contour",
      "type": "function",
      "outputs": None,
      "function": segmentation.active_contour,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.active_contour",
      "parameters": {
        "image": {
          "type": "shape",
          "default": None
        },
        "snake": {
          "type": "shape",
          "default": None
        },
        "alpha": {
          "type": float,
          "default": 0.01
        },
        "beta": {
          "type": float,
          "default": 0.1
        },
        "w_line": {
          "type": float,
          "default": 0.0
        },
        "w_edge": {
          "type": int,
          "default": 1
        },
        "gamma": {
          "type": float,
          "default": 0.01
        },
        "max_px_move": {
          "type": float,
          "default": 1.0
        },
        "max_num_iter": {
          "type": int,
          "default": 2500
        },
        "convergence": {
          "type": float,
          "default": 0.1
        },
        "boundary_condition": {
          "type": str,
          "default": "periodic"
        }
      }
    },
    "skimage.segmentation.chan_vese": {
      "name": "chan_vese",
      "function_name": "chan_vese",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "segmentation"
        },
        "1": {
          "type": "image",
          "name": "phi"
        },
        "0": {
          "type": list,
          "name": "energies"
        }
      },
      "function": segmentation.chan_vese,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.chan_vese",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "mu": {
          "type": float,
          "default": 0.25
        },
        "lambda1": {
          "type": float,
          "default": 1.0
        },
        "lambda2": {
          "type": float,
          "default": 1.0
        },
        "tol": {
          "type": float,
          "default": 0.001
        },
        "max_num_iter": {
          "type": int,
          "default": 500
        },
        "dt": {
          "type": float,
          "default": 0.5
        },
        "init_level_set": {
          "type": str,
          "default": "checkerboard"
        },
        "extended_output": {
          "type": bool,
          "default": False
        }
      }
    },
    "skimage.segmentation.checkerboard_level_set": {
      "name": "checkerboard_level_set",
      "function_name": "checkerboard_level_set",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.checkerboard_level_set,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.checkerboard_level_set",
      "parameters": {
        "image_shape": {
          "type": "image",
          "default": None
        },
        "square_size": {
          "type": int,
          "default": 5
        }
      }
    },
    "skimage.segmentation.clear_border": {
      "name": "clear_border",
      "function_name": "clear_border",
      "type": "function",
      "outputs": None,
      "function": segmentation.clear_border,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.clear_border",
      "parameters": {
        "labels": {
          "type": "image",
          "default": None
        },
        "buffer_size": {
          "type": int,
          "default": 0
        },
        "bgval": {
          "type": int,
          "default": 0
        },
        "mask": {
          "type": "image",
          "default": None
        },
        # "out": {
        #   "type": "NoneType",
        #   "default": None
        # }
      }
    },
    "skimage.segmentation.disk_level_set": {
      "name": "disk_level_set",
      "function_name": "disk_level_set",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.disk_level_set,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.disk_level_set",
      "parameters": {
        "image_shape": {
          "type": tuple,
          "default": None
        },
        "center": {
          "type": (tuple, None),
          "default": None
        },
        "radius": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.segmentation.expand_labels": {
      "name": "expand_labels",
      "function_name": "expand_labels",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.expand_labels,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.expand_labels",
      "parameters": {
        "label_image": {
          "type": "image",
          "default": None
        },
        "distance": {
          "type": int,
          "default": 1
        },
        "spacing": {
          "type": int,
          "default": 1
        }
      }
    },
    "skimage.segmentation.felzenszwalb": {
      "name": "felzenszwalb",
      "function_name": "felzenszwalb",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.felzenszwalb,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "scale": {
          "type": float,
          "default": 1
        },
        "sigma": {
          "type": float,
          "default": 0.8
        },
        "min_size": {
          "type": int,
          "default": 20
        },
        "channel_axis": {
          "type": (int, None),
          "default": -1
        }
      }
    },
    "skimage.segmentation.find_boundaries": {
      "name": "find_boundaries",
      "function_name": "find_boundaries",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.find_boundaries,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries",
      "parameters": {
        "label_img": {
          "type": "image",
          "default": None
        },
        "connectivity": {
          "type": int,
          "default": 1
        },
        "mode": {
          "type": str,
          "default": "thick"
        },
        "background": {
          "type": int,
          "default": 0
        }
      }
    },
    "skimage.segmentation.flood": {
      "name": "flood",
      "function_name": "flood",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.flood,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.flood",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "seed_point": {
          "type": (tuple, int),
          "default": None
        },
        "footprint": {
          "type": "image",
          "default": None
        },
        "connectivity": {
          "type": (int, None),
          "default": None
        },
        "tolerance": {
          "type": (float, None),
          "default": None
        }
      }
    },
    "skimage.segmentation.flood_fill": {
      "name": "flood_fill",
      "function_name": "flood_fill",
      "type": "function",
      "outputs": None,
      "function": segmentation.flood_fill,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.flood_fill",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "seed_point": {
          "type": tuple,
          "default": None
        },
        "new_value": {
          "type": type,
          "default": None
        },
        "footprint": {
          "type": "image",
          "default": None
        },
        "connectivity": {
          "type": (int, None),
          "default": None
        },
        "tolerance": {
          "type": (float, None),
          "default": None
        },
        "in_place": {
          "type": bool,
          "default": False
        }
      }
    },
    "skimage.segmentation.inverse_gaussian_gradient": {
      "name": "inverse_gaussian_gradient",
      "function_name": "inverse_gaussian_gradient",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.inverse_gaussian_gradient,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.inverse_gaussian_gradient",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "alpha": {
          "type": float,
          "default": 100.0
        },
        "sigma": {
          "type": float,
          "default": 5.0
        }
      }
    },
    "skimage.segmentation.join_segmentations": {
      "name": "join_segmentations",
      "function_name": "join_segmentations",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        },
        "1": {
          "type": map,
          "name": "mapping"
        },
        "2": {
          "type": map,
          "name": "mapping2"
        }
      },
      "function": segmentation.join_segmentations,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.join_segmentations",
      "parameters": {
        "s1": {
          "type": "image",
          "default": None
        },
        "s2": {
          "type": "image",
          "default": None
        },
        # "return_mapping": {
        #   "type": "<class 'bool'>",
        #   "default": False
        # }
      }
    },
    "skimage.segmentation.mark_boundaries": {
      "name": "mark_boundaries",
      "function_name": "mark_boundaries",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.mark_boundaries,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.mark_boundaries",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "label_img": {
          "type": "image",
          "default": None
        },
        "color": {
          "type": tuple,
          "default": "(1, 1, 0)"
        },
        "outline_color": {
          "type": (tuple, None),
          "default": None
        },
        "mode": {
          "type": str,
          "default": "outer"
        },
        "background_label": {
          "type": int,
          "default": 0
        }
      }
    },
    "skimage.segmentation.morphological_chan_vese": {
      "name": "morphological_chan_vese",
      "function_name": "morphological_chan_vese",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "segmentation"
        }
      },
      "function": segmentation.morphological_chan_vese,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.morphological_chan_vese",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "num_iter": {
          "type": int,
          "default": 1
        },
        "init_level_set": {
          "type": str,
          "default": "checkerboard"
        },
        "smoothing": {
          "type": int,
          "default": 1
        },
        "lambda1": {
          "type": int,
          "default": 1
        },
        "lambda2": {
          "type": int,
          "default": 1
        },
        "iter_callback": {
          "type": ("function", None),
          "default": "lambda x: None"
        }
      }
    },
    "skimage.segmentation.morphological_geodesic_active_contour": {
      "name": "morphological_geodesic_active_contour",
      "function_name": "morphological_geodesic_active_contour",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "segmentation"
        }
      },
      "function": segmentation.morphological_geodesic_active_contour,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.morphological_geodesic_active_contour",
      "parameters": {
        "gimage": {
          "type": "image",
          "default": None
        },
        "num_iter": {
          "type": int,
          "default": 1
        },
        "init_level_set": {
          "type": (str, "image"),
          "default": "disk"
        },
        "smoothing": {
          "type": int,
          "default": 1
        },
        "threshold": {
          "type": str,
          "default": "auto"
        },
        "balloon": {
          "type": int,
          "default": 0
        },
        "iter_callback": {
          "type": ("function", None),
          "default": "lambda x: None"
        }
      }
    },
    "skimage.segmentation.quickshift": {
      "name": "quickshift",
      "function_name": "quickshift",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.quickshift,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.quickshift",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "ratio": {
          "type": float,
          "default": 1.0
        },
        "kernel_size": {
          "type": float,
          "default": 5
        },
        "max_dist": {
          "type": float,
          "default": 10
        },
        # "return_tree": {
        #   "type": bool,
        #   "default": False
        # },
        "sigma": {
          "type": float,
          "default": 0
        },
        "convert2lab": {
          "type": bool,
          "default": True
        },
        "rng": {
          "type": int,
          "default": 42
        },
        "channel_axis": {
          "type": int,
          "default": -1
        }
      }
    },
    "skimage.segmentation.random_walker": {
      "name": "random_walker",
      "function_name": "random_walker",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "labels"
        }
      },
      "function": segmentation.random_walker,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.random_walker",
      "parameters": {
        "data": {
          "type": "image",
          "default": None
        },
        "labels": {
          "type": "image",
          "default": None
        },
        "beta": {
          "type": float,
          "default": 130
        },
        "mode": {
          "type": str,
          "default": "cg_j"
        },
        "tol": {
          "type": float,
          "default": 0.001
        },
        "copy": {
          "type": bool,
          "default": True
        },
        "return_full_prob": {
          "type": bool,
          "default": False
        },
        "spacing": {
          "type": (tuple, None),
          "default": None
        },
        "prob_tol": {
          "type": float,
          "default": 0.001
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.segmentation.relabel_sequential": {
      "name": "relabel_sequential",
      "function_name": "relabel_sequential",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        },
        "1": {
          "type": map,
          "name": "forward_map"
        },
        "2": {
          "type": map,
          "name": "inverse_map"
        }
      },
      "function": segmentation.relabel_sequential,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.relabel_sequential",
      "parameters": {
        "label_field": {
          "type": "image",
          "default": None
        },
        "offset": {
          "type": int,
          "default": 1
        }
      }
    },
    "skimage.segmentation.slic": {
      "name": "slic",
      "function_name": "slic",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.slic,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "n_segments": {
          "type": int,
          "default": 100
        },
        "compactness": {
          "type": float,
          "default": 10.0
        },
        "max_num_iter": {
          "type": int,
          "default": 10
        },
        "sigma": {
          "type": (float, tuple),
          "default": 0
        },
        "spacing": {
          "type": (tuple, None),
          "default": None
        },
        "convert2lab": {
          "type": (bool, None),
          "default": None
        },
        "enforce_connectivity": {
          "type": bool,
          "default": True
        },
        "min_size_factor": {
          "type": float,
          "default": 0.5
        },
        "max_size_factor": {
          "type": float,
          "default": 3
        },
        "slic_zero": {
          "type": bool,
          "default": False
        },
        "start_label": {
          "type": int,
          "default": 1
        },
        "mask": {
          "type": "image",
          "default": None
        },
        "channel_axis": {
          "type": (int, None),
          "default": -1
        }
      }
    },
    "skimage.segmentation.watershed": {
      "name": "watershed",
      "function_name": "watershed",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": segmentation.watershed,
      "help": "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "markers": {
          "type": "image",
          "default": None
        },
        "connectivity": {
          "type": int,
          "default": 1
        },
        "offset": {
          "type": (tuple, None),
          "default": None
        },
        "mask": {
          "type": "image",
          "default": None
        },
        "compactness": {
          "type": float,
          "default": 0
        },
        "watershed_line": {
          "type": bool,
          "default": False
        }
      }
    }
  },
  "transform": {
    "skimage.transform.downscale_local_mean": {
      "name": "downscale_local_mean",
      "function_name": "downscale_local_mean",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.downscale_local_mean,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.downscale_local_mean",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "factors": {
          "type": tuple,
          "default": None
        },
        "cval": {
          "type": float,
          "default": 0
        },
        "clip": {
          "type": bool,
          "default": True
        }
      }
    },
    # "estimate_transform": {
    #   "name": "estimate_transform",
    #   "function_name": "estimate_transform",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.estimate_transform,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.estimate_transform",
    #   "parameters": {
    #     "ttype": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "src": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "dst": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "args": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "kwargs": {
    #       "type": "type",
    #       "default": None
    #     }
    #   }
    # },
    "skimage.transform.frt2": {
      "name": "frt2",
      "function_name": "frt2",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.frt2,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.frt2",
      "parameters": {
        "a": {
          "type": "image",
          "default": None
        }
      }
    },
    # "hough_circle": {
    #   "name": "hough_circle",
    #   "function_name": "hough_circle",
    #   "type": "function",
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "hspace"
    #     },
    #     "1": {
    #       "type": (list, None),
    #       "name": "radii"
    #     },
    #   },
    #   "function": transform.hough_circle,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_circle",
    #   "parameters": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     },
    #     "radius": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "normalize": {
    #       "type": bool,
    #       "default": True
    #     },
    #     "full_output": {
    #       "type": bool,
    #       "default": False
    #     }
    #   }
    # },
    # "hough_circle_peaks": {
    #   "name": "hough_circle_peaks",
    #   "function_name": "hough_circle_peaks",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.hough_circle_peaks,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_circle_peaks",
    #   "parameters": {
    #     "hspaces": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "radii": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "min_xdistance": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "min_ydistance": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "threshold": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "num_peaks": {
    #       "type": float,
    #       "default": Infinity
    #     },
    #     "total_num_peaks": {
    #       "type": float,
    #       "default": Infinity
    #     },
    #     "normalize": {
    #       "type": bool,
    #       "default": False
    #     }
    #   }
    # },
    # "hough_ellipse": {
    #   "name": "hough_ellipse",
    #   "function_name": "hough_ellipse",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.hough_ellipse,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_ellipse",
    #   "parameters": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     },
    #     "threshold": {
    #       "type": int,
    #       "default": 4
    #     },
    #     "accuracy": {
    #       "type": float,
    #       "default": 1
    #     },
    #     "min_size": {
    #       "type": int,
    #       "default": 4
    #     },
    #     "max_size": {
    #       "type": (int, None),
    #       "default": None
    #     }
    #   }
    # },
    # "hough_line": {
    #   "name": "hough_line",
    #   "function_name": "hough_line",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.hough_line,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_line",
    #   "parameters": {
    #     "image": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "theta": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "hough_line_peaks": {
    #   "name": "hough_line_peaks",
    #   "function_name": "hough_line_peaks",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.hough_line_peaks,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_line_peaks",
    #   "parameters": {
    #     "hspace": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "angles": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "dists": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "min_distance": {
    #       "type": int,
    #       "default": 9
    #     },
    #     "min_angle": {
    #       "type": int,
    #       "default": 10
    #     },
    #     "threshold": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "num_peaks": {
    #       "type": float,
    #       "default": Infinity
    #     }
    #   }
    # },
    "skimage.transform.ifrt2": {
      "name": "ifrt2",
      "function_name": "ifrt2",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.ifrt2,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.ifrt2",
      "parameters": {
        "a": {
          "type": "image",
          "default": None
        }
      }
    },
    # "integral_image": {
    #   "name": "integral_image",
    #   "function_name": "integral_image",
    #   "type": "function",
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "integral_image"
    #     }
    #   },
    #   "function": transform.integral_image,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.integral_image",
    #   "parameters": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     },
    #     "dtype": {
    #       "type": (type, None),
    #       "default": None
    #     }
    #   }
    # },
    # "integrate": {
    #   "name": "integrate",
    #   "function_name": "integrate",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.integrate,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.integrate",
    #   "parameters": {
    #     "ii": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "start": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "end": {
    #       "type": "type",
    #       "default": None
    #     }
    #   }
    # },
    "skimage.transform.iradon": {
      "name": "iradon",
      "function_name": "iradon",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.iradon,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.iradon",
      "parameters": {
        "radon_image": {
          "type": "image",
          "default": None
        },
        "theta": {
          "type": (tuple, None),
          "default": None
        },
        "output_size": {
          "type": (tuple, None),
          "default": None
        },
        "filter_name": {
          "type": str,
          "default": "ramp"
        },
        "interpolation": {
          "type": str,
          "default": "linear"
        },
        "circle": {
          "type": bool,
          "default": True
        },
        "preserve_range": {
          "type": bool,
          "default": True
        }
      }
    },
    "skimage.transform.iradon_sart": {
      "name": "iradon_sart",
      "function_name": "iradon_sart",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.iradon_sart,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.iradon_sart",
      "parameters": {
        "radon_image": {
          "type": "image",
          "default": None
        },
        "theta": {
          "type": (tuple, None),
          "default": None
        },
        "image": {
          "type": ("image", None),
          "default": None
        },
        "projection_shifts": {
          "type": (tuple, None),
          "default": None
        },
        "clip": {
          "type": (tuple, None),
          "default": None
        },
        "relaxation": {
          "type": float,
          "default": 0.15
        },
        "dtype": {
          "type": (type, None),
          "default": None
        }
      }
    },
    # "matrix_transform": {
    #   "name": "matrix_transform",
    #   "function_name": "matrix_transform",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.matrix_transform,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.matrix_transform",
    #   "parameters": {
    #     "coords": {
    #       "type": "image",
    #       "default": None
    #     },
    #     "matrix": {
    #       "type": matrix,
    #       "default": None
    #     }
    #   }
    # },
    # "order_angles_golden_ratio": {
    #   "name": "order_angles_golden_ratio",
    #   "function_name": "order_angles_golden_ratio",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.order_angles_golden_ratio,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.order_angles_golden_ratio",
    #   "parameters": {
    #     "theta": {
    #       "type": "type",
    #       "default": None
    #     }
    #   }
    # },
    # "probabilistic_hough_line": {
    #   "name": "probabilistic_hough_line",
    #   "function_name": "probabilistic_hough_line",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.probabilistic_hough_line,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.probabilistic_hough_line",
    #   "parameters": {
    #     "image": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "threshold": {
    #       "type": int,
    #       "default": 10
    #     },
    #     "line_length": {
    #       "type": int,
    #       "default": 50
    #     },
    #     "line_gap": {
    #       "type": int,
    #       "default": 10
    #     },
    #     "theta": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "rng": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    "skimage.transform.pyramid_expand": {
      "name": "pyramid_expand",
      "function_name": "pyramid_expand",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.pyramid_expand,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.pyramid_expand",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "upscale": {
          "type": int,
          "default": 2
        },
        "sigma": {
          "type": "NoneType",
          "default": None
        },
        "order": {
          "type": int,
          "default": 1
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": int,
          "default": 0
        },
        "preserve_range": {
          "type": bool,
          "default": False
        },
        "channel_axis": {
          "type": "NoneType",
          "default": None
        }
      }
    },
    # "pyramid_gaussian": {
    #   "name": "pyramid_gaussian",
    #   "function_name": "pyramid_gaussian",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.pyramid_gaussian,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.pyramid_gaussian",
    #   "parameters": {
    #     "image": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "max_layer": {
    #       "type": int,
    #       "default": -1
    #     },
    #     "downscale": {
    #       "type": int,
    #       "default": 2
    #     },
    #     "sigma": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "order": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "mode": {
    #       "type": str,
    #       "default": "reflect"
    #     },
    #     "cval": {
    #       "type": int,
    #       "default": 0
    #     },
    #     "preserve_range": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "channel_axis": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "pyramid_laplacian": {
    #   "name": "pyramid_laplacian",
    #   "function_name": "pyramid_laplacian",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.pyramid_laplacian,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.pyramid_laplacian",
    #   "parameters": {
    #     "image": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "max_layer": {
    #       "type": int,
    #       "default": -1
    #     },
    #     "downscale": {
    #       "type": int,
    #       "default": 2
    #     },
    #     "sigma": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "order": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "mode": {
    #       "type": str,
    #       "default": "reflect"
    #     },
    #     "cval": {
    #       "type": int,
    #       "default": 0
    #     },
    #     "preserve_range": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "channel_axis": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    # "pyramid_reduce": {
    #   "name": "pyramid_reduce",
    #   "function_name": "pyramid_reduce",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.pyramid_reduce,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.pyramid_reduce",
    #   "parameters": {
    #     "image": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "downscale": {
    #       "type": int,
    #       "default": 2
    #     },
    #     "sigma": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "order": {
    #       "type": int,
    #       "default": 1
    #     },
    #     "mode": {
    #       "type": str,
    #       "default": "reflect"
    #     },
    #     "cval": {
    #       "type": int,
    #       "default": 0
    #     },
    #     "preserve_range": {
    #       "type": bool,
    #       "default": False
    #     },
    #     "channel_axis": {
    #       "type": "NoneType",
    #       "default": None
    #     }
    #   }
    # },
    "skimage.transform.radon": {
      "name": "radon",
      "function_name": "radon",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "sinogram"
        }
      },
      "function": transform.radon,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.radon",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "theta": {
          "type": (tuple, None),
          "default": None
        },
        "circle": {
          "type": bool,
          "default": True
        },
        "preserve_range": {
          "type": bool,
          "default": False
        }
      }
    },
    "skimage.transform.rescale": {
      "name": "rescale",
      "function_name": "rescale",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.rescale,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rescale",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "scale": {
          "type": (float, tuple),
          "default": None
        },
        "order": {
          "type": (int, None),
          "default": None
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": int,
          "default": 0
        },
        "clip": {
          "type": bool,
          "default": True
        },
        "preserve_range": {
          "type": bool,
          "default": False
        },
        "anti_aliasing": {
          "type": (bool, None),
          "default": None
        },
        "anti_aliasing_sigma": {
          "type": (float, tuple, None),
          "default": None
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.transform.resize": {
      "name": "resize",
      "function_name": "resize",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.resize,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "output_shape": {
          "type": (tuple, None),
          "default": None
        },
        "order": {
          "type": (int, None),
          "default": None
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": int,
          "default": 0
        },
        "clip": {
          "type": bool,
          "default": True
        },
        "preserve_range": {
          "type": bool,
          "default": False
        },
        "anti_aliasing": {
          "type": (bool, None),
          "default": None
        },
        "anti_aliasing_sigma": {
          "type": (float, tuple, None),
          "default": None
        }
      }
    },
    "skimage.transform.resize_local_mean": {
      "name": "resize_local_mean",
      "function_name": "resize_local_mean",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.resize_local_mean,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize_local_mean",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "output_shape": {
          "type": (tuple, None),
          "default": None
        },
        "grid_mode": {
          "type": bool,
          "default": True
        },
        "preserve_range": {
          "type": bool,
          "default": False
        },
        "channel_axis": {
          "type": (int, None),
          "default": None
        }
      }
    },
    "skimage.transform.rotate": {
      "name": "rotate",
      "function_name": "rotate",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.rotate,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "angle": {
          "type": float,
          "default": 0.
        },
        "resize": {
          "type": bool,
          "default": False
        },
        "center": {
          "type": (tuple, None),
          "default": None
        },
        "order": {
          "type": int,
          "default": 0
        },
        "mode": {
          "type": str,
          "default": "constant"
        },
        "cval": {
          "type": int,
          "default": 0
        },
        "clip": {
          "type": bool,
          "default": True
        },
        "preserve_range": {
          "type": bool,
          "default": False
        }
      }
    },
    "skimage.transform.swirl": {
      "name": "swirl",
      "function_name": "swirl",
      "type": "function",
      "outputs": {
        "0": {
          "type": "image",
          "name": "image"
        }
      },
      "function": transform.swirl,
      "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.swirl",
      "parameters": {
        "image": {
          "type": "image",
          "default": None
        },
        "center": {
          "type": (tuple, None),
          "default": None
        },
        "strength": {
          "type": int,
          "default": 1
        },
        "radius": {
          "type": int,
          "default": 100
        },
        "rotation": {
          "type": int,
          "default": 0
        },
        "output_shape": {
          "type":(tuple, None),       
          "default": None
        },
        "order": {
          "type": int,
          "default": 0
        },
        "mode": {
          "type": str,
          "default": "reflect"
        },
        "cval": {
          "type": int,
          "default": 0
        },
        "clip": {
          "type": bool,
          "default": True
        },
        "preserve_range": {
          "type": bool,
          "default": False
        }
      }
    },
    # "warp": {
    #   "name": "warp",
    #   "function_name": "warp",
    #   "type": "function",
    #   "outputs": {
    #     "0": {
    #       "type": "image",
    #       "name": "image"
    #     }
    #   },
    #   "function": transform.warp,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp",
    #   "parameters": {
    #     "image": {
    #       "type": "image",
    #       "default": None
    #     },
    #     "inverse_map": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "map_args": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "output_shape": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "order": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "mode": {
    #       "type": str,
    #       "default": "constant"
    #     },
    #     "cval": {
    #       "type": float,
    #       "default": 0.0
    #     },
    #     "clip": {
    #       "type": bool,
    #       "default": True
    #     },
    #     "preserve_range": {
    #       "type": bool,
    #       "default": False
    #     }
    #   }
    # },
    # "warp_coords": {
    #   "name": "warp_coords",
    #   "function_name": "warp_coords",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.warp_coords,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp_coords",
    #   "parameters": {
    #     "coord_map": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "shape": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "dtype": {
    #       "type": "type",
    #       "default": "<class 'numpy.float64'>"
    #     }
    #   }
    # },
    # "warp_polar": {
    #   "name": "warp_polar",
    #   "function_name": "warp_polar",
    #   "type": "function",
    #   "outputs": None,
    #   "function": transform.warp_polar,
    #   "help": "https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp_polar",
    #   "parameters": {
    #     "image": {
    #       "type": "type",
    #       "default": None
    #     },
    #     "center": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "radius": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "output_shape": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "scaling": {
    #       "type": str,
    #       "default": "linear"
    #     },
    #     "channel_axis": {
    #       "type": "NoneType",
    #       "default": None
    #     },
    #     "kwargs": {
    #       "type": "type",
    #       "default": None
    #     }
    #   }
    # }
  }
}