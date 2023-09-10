#include <vits.h>
#include <stdio.h>
#include <assert.h>

int main(int argc, char ** argv) {
    vits_model * model = vits_model_load_from_file("model.onnx");
    assert(model != nullptr);

    auto result = vits_model_process(model, "phonemes");
    printf("Generated: %d bytes of audio\n", result->size);

    vits_free_result(result);
    vits_free_model(model);
    return 0;
}
