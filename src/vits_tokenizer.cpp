//
// Created by Maximiliano Levi on 11/12/23.
//

#include "include/vits_tokenizer.h"
#include "include/common.h"

vits_tokenizer::vits_tokenizer() {

}
vits_tokenizer::~vits_tokenizer() {

}

std::unique_ptr<vits_tokenizer> vits_tokenizer::load(std::ifstream& file) {
    // Read the vocabulary size
    auto tokenizer = std::make_unique<vits_tokenizer>();
    uint32_t vocab_size = read_number(file);
    printf("Loading tokenizer with %u tokens\n", vocab_size);
    for(uint32_t i = 0; i < vocab_size; ++i) {
        // Read key
        uint32_t key_len = read_number(file);
        std::vector<char> key_bytes(key_len);
        file.read(key_bytes.data(), key_len);
        std::string key(key_bytes.begin(), key_bytes.end());

        // Read value
        uint32_t value = read_number(file);
        tokenizer->vocab[key] = value;
    }

    // Read tokenizer options
    tokenizer->add_blank = read_number(file);
    tokenizer->normalize = read_number(file);

    // Read pad token and unk token
    uint32_t pad_token_len = read_number(file);
    std::vector<char> pad_token_bytes(pad_token_len);
    file.read(pad_token_bytes.data(), pad_token_len);
    tokenizer->pad_token = std::string(pad_token_bytes.begin(), pad_token_bytes.end());

    uint32_t unk_token_len = read_number(file);
    std::vector<char> unk_token_bytes(unk_token_len);
    file.read(unk_token_bytes.data(), unk_token_len);
    tokenizer->unk_token = std::string(unk_token_bytes.begin(), unk_token_bytes.end());
    printf("Loaded tokenizer with %lu tokens\n", tokenizer->vocab.size());
    return std::move(tokenizer);
}

std::string vits_tokenizer::normalize_text(const std::string& input_string) {
    std::string filtered_text = "";
    size_t i = 0;
    while (i < input_string.length()) {
        bool found_match = false;
        for (const auto& word : vocab) {
            if (input_string.substr(i, word.first.length()) == word.first) {
                filtered_text += word.first;
                i += word.first.length();
                found_match = true;
                break;
            }
        }

        if (!found_match) {
            filtered_text += std::tolower(input_string[i]);
            i++;
        }
    }
    return filtered_text;
}

std::string vits_tokenizer::prepare_for_tokenization(const std::string& text, bool is_split_into_words, bool normalize) {
    std::string processed_text = text;

    if (normalize) {
        processed_text = normalize_text(processed_text);

        processed_text.erase(std::remove_if(processed_text.begin(), processed_text.end(),
                                            [this](char c) { return vocab.find(std::string(1, c)) == vocab.end(); }), processed_text.end());
    }

    return {processed_text, {}};
}

std::vector<int32_t> vits_tokenizer::tokenize(const std::string& text) {
    auto normalized_text = this->prepare_for_tokenization(text, false, normalize);
    printf("Normalized text: %s\n", normalized_text.c_str());

    std::vector<std::string> token_strs;
    for (char c : normalized_text) {
        token_strs.push_back(std::string(1, c));
    }

    if (add_blank) {
        std::vector<std::string> interspersed(token_strs.size() * 2 + 1, pad_token);
        for (size_t i = 0; i < token_strs.size(); ++i) {
            interspersed[i * 2 + 1] = token_strs[i];
        }
        token_strs = interspersed;
    }
    std::vector<int32_t> tokens;
    for (const auto& token : token_strs) {
        tokens.push_back(convert_token_to_id(token));
    }
    return tokens;
}

int vits_tokenizer::convert_token_to_id(const std::string& token) {
    auto it = vocab.find(token);
    if (it != vocab.end()) {
        return it->second;
    } else {
        auto unk_it = vocab.find(unk_token);
        return unk_it != vocab.end() ? unk_it->second : -1;
    }
}