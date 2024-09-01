#include <vits.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>

struct WAVHeader {
    char riff_header[4];         // Contains "RIFF"
    int wav_size;                // Size of the WAV file
    char wave_header[4];         // Contains "WAVE"
    char fmt_header[4];          // Contains "fmt " (with a space after fmt)
    int fmt_chunk_size;          // Should be 16 for PCM format
    short audio_format;          // Should be 1 for PCM format
    short num_channels;
    int sample_rate;
    int byte_rate;               // Number of bytes per second. sample_rate * num_channels * Bytes Per Sample
    short sample_alignment;      // num_channels * Bytes Per Sample
    short bit_depth;             // Number of bits per sample
    char data_header[4];         // Contains "data"
    int data_bytes;              // Number of bytes in data. Number of samples * num_channels * sample byte size
};

bool write_wav(std::string path, float* samples, size_t size) {
    WAVHeader wav_header;
    int sample_rate = 16000;
    int num_channels = 1;
    int bit_depth = 16;

    std::vector<short> pcm_samples(size);
    for (size_t i = 0; i < size; ++i) {
        pcm_samples[i] = static_cast<short>(std::max(-1.0f, std::min(1.0f, samples[i])) * 32767);
    }

    // Open file
    std::ofstream file(path, std::ios::binary);

    // Write the WAV header
    memcpy(wav_header.riff_header, "RIFF", 4);
    memcpy(wav_header.wave_header, "WAVE", 4);
    memcpy(wav_header.fmt_header, "fmt ", 4);
    wav_header.fmt_chunk_size = 16;
    wav_header.audio_format = 1;
    wav_header.num_channels = num_channels;
    wav_header.sample_rate = sample_rate;
    wav_header.byte_rate = sample_rate * num_channels * (bit_depth / 8);
    wav_header.sample_alignment = num_channels * (bit_depth / 8);
    wav_header.bit_depth = bit_depth;
    memcpy(wav_header.data_header, "data", 4);
    wav_header.data_bytes = pcm_samples.size() * (bit_depth / 8);
    wav_header.wav_size = 4 + (8 + wav_header.fmt_chunk_size) + (8 + wav_header.data_bytes);

    // Write header to file
    file.write(reinterpret_cast<const char*>(&wav_header), sizeof(WAVHeader));

    // Write audio samples
    file.write(reinterpret_cast<const char*>(pcm_samples.data()), wav_header.data_bytes);

    // Close file
    file.close();

    std::cout << "WAV file '" << path << "' has been written" << std::endl;
    return true;
}

struct VITSParams {
    int n_threads                 = -1;
    std::string model_path        = "./scripts/vits-spanish.ggml";
    std::string phrase            = "Cada amanecer trae consigo nuevas oportunidades para crecer y aprender";
    std::string output_path       = "./output.wav";
    int64_t seed                  = -1;
};

void print_usage(int argc, char** argv) {
    printf("usage: %s [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  -t, --threads N                    number of threads to use during computation (default: -1).\n");
    printf("                                     If threads <= 0, then threads will be set to the number of CPU physical cores\n");
    printf("  -m, --model   MODEL                path to model\n");
    printf("  -p, --phrase  PHRASE               phrase to say\n");
    printf("  -o, --output  OUTPUT_PATH          path to write result image to (default: ./output.wav)\n");
    printf("  -s SEED, --seed SEED               RNG seed (default: -1, use random seed for < 0)\n");
}

void parse_args(int argc, char** argv, VITSParams& params) {
    bool invalid_arg = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
	arg = argv[i];
	if (arg == "-t" || arg == "--threads") {
	    if (++i >= argc) {
		invalid_arg = true;
		break;
	    }
	    params.n_threads = std::stoi(argv[i]);
	} else if (arg == "-m" || arg == "--model") {
	    if (++i >= argc) {
		invalid_arg = true;
		break;
	    }
	    params.model_path = argv[i];
	} else if (arg == "-o" || arg == "--output") {
	    if (++i >= argc) {
                invalid_arg = true;
		break;
	    }
	    params.output_path = argv[i];
	} else if (arg == "-p" || arg == "--phrase") {
	    if (++i >= argc) {
                invalid_arg = true;
		break;
	    }
	    params.phrase = argv[i];
	} else if (arg == "-s" || arg == "--seed") {
	    if (++i >= argc) {
		invalid_arg = true;
		break;
	    }
	    params.seed = std::stoll(argv[i]);
	} else if (arg == "-h" || arg == "--help") {
	    print_usage(argc, argv);
	    exit(0);
	} else {
	    fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
	    print_usage(argc, argv);
	    exit(1);
	}
    }
    if (invalid_arg) {
	fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
	print_usage(argc, argv);
	exit(1);
    }
    if (params.n_threads <= 0) {
	unsigned int n_threads = std::thread::hardware_concurrency();
	params.n_threads = (n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4);
    }
    if (params.seed < 0) {
	srand((int)time(NULL));
	params.seed = rand();
    }
}

int main(int argc, char** argv) {
    VITSParams params;
    parse_args(argc, argv, params);
    
    vits_model * model = vits_model_load_from_file(params.model_path.c_str());
    assert(model != nullptr);

    //rng.seed(params.seed);

    auto result = vits_model_process(model, params.phrase.c_str());
    //auto result = vits_model_process(model, params.phrase.c_str(), params.n_threads);
    if (result.size > 0) {
        printf("Generated: %d samples of audio %f %f %f\n", result.size, result.data[0], result.data[1],
               result.data[2]);
        printf("Wrote to file: %s\n", write_wav(params.output_path, result.data, result.size) ? "true" : "false");
    }
    vits_free_result(result);
    vits_free_model(model);
    return 0;
}
