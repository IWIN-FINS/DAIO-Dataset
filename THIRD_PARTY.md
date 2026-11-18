# Third-party sources

| Software | Source revision | Included license |
|---|---|---|
| AirIO | [Air-IO/Air-IO at 7214e8a](https://github.com/Air-IO/Air-IO/tree/7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73) | [AirIO LICENSE](new-dataset/baselines/AirIO/LICENSE) |
| AirIMU | [haleqiu/AirIMU at 119fe7c](https://github.com/haleqiu/AirIMU/tree/119fe7ce229e321b3890282909a183aefad61dfb) | [AirIMU LICENSE](new-dataset/baselines/AirIMU/LICENSE) |
| DeepVL | [ntnu-arl/DeepVL at 776383e](https://github.com/ntnu-arl/DeepVL/tree/776383e9df6098e223304eb0362f7e4d7b198875) | [DeepVL LICENSE](new-dataset/deepvl/vendor/DeepVL/LICENSE) |
| ReAqROVIO | [ntnu-arl/reaqrovio at 7fc95c1](https://github.com/ntnu-arl/reaqrovio/tree/7fc95c1fe97c8a4e89a1477d5699807718f37e43) | [ReAqROVIO LICENSE and component notices](new-dataset/deepvl/vendor/ReAqROVIO/LICENSE) |

The planar DeepVL observation and calibrated inertial/depth filter are provided in `new-dataset/deepvl/` and `new-dataset/deepvl_filter_runtime.py`. The three-dimensional AirIO adaptation uses past-only velocity predictions and the ROV mixed reference described in the README. Third-party notices apply to their respective components.
