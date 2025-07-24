# Generate runner
ruby Middlewares\Third_Party\Unity\auto\generate_test_runner.rb Tests\test_bme280.c Tests\test_runners\test_runner_bme280.c

# Compile
gcc Tests\test_main.c Tests\test_runners\test_runner_bme280.c Middlewares\Third_Party\Unity\src\unity.c -ICore\Inc -ITasks\Sensors -IMiddlewares\Third_Party\Unity\src -o test_bme280.exe

# Run
.\test_bme280.exe