//
// Created by Maximiliano Levi on 11/12/23.
//

#include "include/vits_tokenizer.h"
#include "include/common.h"
#include "include/debug.h"

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

std::vector<int32_t> vits_tokenizer::tokenize_fast(const std::string& input_string) {
    std::vector<int32_t> tokens;
    size_t i = 0;
    while (i < input_string.length()) {
        bool found_match = false;
        for (const auto& word : vocab) {
            if (input_string.substr(i, word.first.length()) == word.first) {
                tokens.push_back(word.second);
                i += word.first.length();
                found_match = true;
                break;
            }
        }

        if (!found_match) {
            printf("Could not find match for %s\n", input_string.substr(i, 1).c_str());
            i++;
        }
    }
    printf("Tokenized text: '%s'\n", input_string.c_str());
    return tokens;
}

std::string vits_tokenizer::normalize_text(const std::string& input_string) {
    std::string filtered_text = "";
    size_t i = 0;
    while (i < input_string.length()) {
        bool found_match = false;
        for (const auto& word : vocab) {
            if (input_string.substr(i, word.first.length()) == word.first) {
                filtered_text += word.first;
                printf("Matched %s for %s\n", word.first.c_str(), filtered_text.c_str());
                i += word.first.length();
                found_match = true;
                break;
            }
        }

        if (!found_match) {
            //filtered_text += input_string[i];
            i++;
        }
    }
    return filtered_text;
}

std::string vits_tokenizer::prepare_for_tokenization(const std::string& text, bool is_split_into_words, bool normalize) {
    std::string processed_text = text;
    std::transform(processed_text.begin(), processed_text.end(), processed_text.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (normalize) {
        processed_text = normalize_text(processed_text);
        /*
        processed_text.erase(
                std::remove_if(processed_text.begin(), processed_text.end(),
                                            [this](char c) {
                    return vocab.find(std::string(1, c)) == vocab.end();
                }), processed_text.end()
        );*/
    }
    return {processed_text, {}};
}

std::vector<int32_t> vits_tokenizer::tokenize(const std::string& text) {
    //auto normalized_text = this->prepare_for_tokenization(text, false, normalize);
    //printf("Normalized text: %s\n", normalized_text.c_str());
/*
    std::vector<std::string> token_strs;
    for (char c : normalized_text) {
        token_strs.push_back(std::string(1, c));
    }
*/
    std::string processed_text = text;
    std::transform(processed_text.begin(), processed_text.end(), processed_text.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    std::vector<int32_t> tokens = tokenize_fast(processed_text);
    std::vector<int32_t> tokens_final;
    if (add_blank) {
        std::vector<int32_t> interspersed(tokens.size() * 2 + 1, vocab[pad_token]);
        for (size_t i = 0; i < tokens.size(); ++i) {
            interspersed[i * 2 + 1] = tokens[i];
        }
        tokens_final = interspersed;
    }
    return tokens_final;
}

int vits_tokenizer::convert_token_to_id(const std::string& token) {
    auto it = vocab.find(token);
    if (it != vocab.end()) {
        return it->second;
    } else {
        ASSERT(false, ("Token " + token + " not found in vocab").c_str());
        auto unk_it = vocab.find(unk_token);
        return unk_it != vocab.end() ? unk_it->second : -1;
    }
}