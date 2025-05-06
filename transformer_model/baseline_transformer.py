from lips.benchmark.powergridBenchmark import PowerGridBenchmark
import os
from lips.augmented_simulators.tensorflow_models.transformer import SimpNet
from lips.dataset.scaler import StandardScaler


if __name__ == "__main__":
    
  CUR_PATH = os.getcwd()
  USE_CASE = "lips_idf_2023"
  BENCH_CONFIG_PATH = os.path.join(CUR_PATH, "configs", "benchmarks", "lips_idf_2023.ini")
  DATA_PATH = os.path.join(CUR_PATH, "input_data_local", USE_CASE)
  TRAINED_MODELS = os.path.join(CUR_PATH, "input_data_local", "trained_models")
  LOG_PATH = "logs.log"

  benchmark_kwargs = {"attr_x": ("prod_p", "prod_v", "load_p", "load_q"),
                      "attr_y": ("a_or", "a_ex", "p_or", "p_ex", "v_or", "v_ex"),
                      "attr_tau": ("line_status", "topo_vect"),
                      "attr_physics": None}

  benchmark = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                config_path=BENCH_CONFIG_PATH,
                                benchmark_name="Benchmark_competition",
                                load_data_set=True,
                                log_path=LOG_PATH,
                                **benchmark_kwargs)

  SIM_CONFIG_PATH = os.path.join("configs", "simulators", "tf_fc.ini")

  tf_fc = SimpNet(name="tf_fc",
                          bench_config_path=BENCH_CONFIG_PATH,
                          bench_config_name="Benchmark_competition",
                          bench_kwargs=benchmark_kwargs,
                          sim_config_path=SIM_CONFIG_PATH,
                          sim_config_name="DEFAULT",
                          scaler=StandardScaler,
                          log_path=LOG_PATH,
                  lr= 1e-4)

  # LOAD_PATH = os.path.join(TRAINED_MODELS, USE_CASE)
  # tf_fc.restore(path=LOAD_PATH)

  tf_fc.train(train_dataset=benchmark.train_dataset,
              val_dataset=benchmark.val_dataset,
              epochs=1)
  tf_fc.summary()

  SAVE_PATH = os.path.join(CUR_PATH, "input_data_local", "trained_models", USE_CASE, "tf_fc_transformer")

  try:
    os.mkdir(SAVE_PATH)

  except OSError as error:
    print("Directory already exists")

  tf_fc.save(SAVE_PATH)
  print("Model saved at: ", SAVE_PATH, " with name: ", tf_fc.name)