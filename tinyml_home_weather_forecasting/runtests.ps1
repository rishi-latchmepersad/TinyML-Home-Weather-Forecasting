# PowerShell script to compile and run Unity tests using WinLibs (GCC)

# Define source files
$testMain     = "Tests\test_main.c"
$testBME280   = "Tests\test_bme280.c"
$unitySrc     = "Middlewares\Third_Party\Unity\src\unity.c"
$freertosStub = "Tests\stubs\FreeRTOS_stub.c"

# Include directories (no .h files directly)
$includes = @(
    "-ITests\stubs",
    "-ICore\Inc",
    "-ITasks\Sensors",
    "-IDrivers\BME280",
    "-IMiddlewares\Third_Party\Unity\src",
#    "-IMiddlewares\Third_Party\FreeRTOS\Source\include",
#    "-IMiddlewares\Third_Party\FreeRTOS\Source\portable\GCC\ARM_CM7\r0p1",
  #  "-IMiddlewares\Third_Party\FreeRTOS\Source\CMSIS_RTOS_V2",
    "-I."
)

# Compiler flags
$flags = @(
    "-DUNITY_INCLUDE_CONFIG_H",
    "-DUNIT_TESTING",  # Critical for stub activation
    "-D_WIN32",
    "-Wall",
    "-Wextra",
    "-g",
    "-std=gnu11"
)

# Output binary
$output = "test_bme280.exe"

# Build command
$gccCmd = @(
    "gcc",
    $testMain,
    $testBME280,
    $freertosStub,
    $unitySrc
) + $includes + $flags + "-o" + $output

# Join and run
$cmd = $gccCmd -join " "

Write-Host "Compiling tests..."
Invoke-Expression $cmd

if (Test-Path $output) {
    Write-Host "`nRunning tests..."
    & ".\$output"
} else {
    Write-Host "❌ Build failed. Test binary not created."
}
