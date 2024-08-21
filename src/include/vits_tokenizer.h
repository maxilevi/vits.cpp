//
// Created by Maximiliano Levi on 11/12/23.
//

#ifndef VITS_VITS_TOKENIZER_H
#define VITS_VITS_TOKENIZER_H

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

class vits_tokenizer {
public:
    vits_tokenizer();
    ~vits_tokenizer();
    std::string normalize_text(const std::string& input_string);
    std::string prepare_for_tokenization(const std::string& text, bool is_split_into_words = false, bool normalize = false);
    std::vector<int32_t> tokenize(const std::string& text);
    std::vector<int32_t> tokenize_fast(const std::string& input_string);
    std::string convert_tokens_to_string(const std::vector<std::string>& tokens);
    int convert_token_to_id(const std::string& token);

    static std::unique_ptr<vits_tokenizer> load(std::istream& file);
    void set_phonetic();

private:
    std::unordered_map<std::string, int32_t> vocab;
    bool add_blank;
    bool normalize;
    bool phonetic = false;
    std::string pad_token;
    std::string unk_token;
};

#endif //VITS_VITS_TOKENIZER_H
