//
// Created by Maximiliano Levi on 11/12/23.
//

#include "include/vits_tokenizer.h"
#include "include/common.h"
#include "include/debug.h"


#ifdef VITS_ESPEAK
#include <cstring>
#include <espeak-ng/speak_lib.h>    
#endif

vits_tokenizer::vits_tokenizer() {

}
vits_tokenizer::~vits_tokenizer() {

}

std::unique_ptr<vits_tokenizer> vits_tokenizer::load(std::istream& file) {
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
            //printf("Could not find match for %s\n", input_string.substr(i, 1).c_str());
            i++;
        }
    }
    //printf("Tokenized text: '%s'\n", input_string.c_str());
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

#ifdef VITS_ESPEAK
static bool init_espeak = false;
std::vector<char> convert_to_phonetic(char* b, char* e)
{
    std::vector<char> ret;
    ret.reserve(2*std::distance(b,e));
    while(b != e)
    {
	const std::size_t pos = std::basic_string_view(b,e).find_first_of("!\\,.:;?");
	if(pos != std::string::npos)
	{
	    const char c = b[pos];
	    b[pos] = 0;
	    const char* t = b;
	    const char* rb = espeak_TextToPhonemes((const void**)&t, espeakCHARS_8BIT, 2);
	    ret.insert(ret.end(), rb, rb+std::strlen(rb));
	    ret.push_back(c);
	    b+=(pos+1);
	    if(c == '.' && b[0] == '.' && b[1] == '.')
	    {
		ret.push_back('.');
		ret.push_back('.');
		b+=2;
	    }
	    if(b != e)
		ret.push_back(' ');
		
	}
	else
	{
	    const char* t = b;
	    const char* rb = espeak_TextToPhonemes((const void**)&t, espeakCHARS_8BIT, 2);
	    ret.insert(ret.end(), rb, rb+std::strlen(rb));
	    b=e;
	}
    }
    return ret;
}

void vits_tokenizer::set_phonetic()
{
    if(!init_espeak)
    {
	espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);
	espeak_VOICE voice{nullptr, nullptr, nullptr, 0, 0};
	voice.languages="en-us";
	espeak_ERROR result = espeak_SetVoiceByProperties(&voice);
	ASSERT(result == EE_OK, "Espeak did not correctly initialize");
	init_espeak = true;
	//we have no espeak_Terminate();
    }
    phonetic = true;
}
#else
void vits_tokenizer::set_phonetic()
{
    ASSERT(false, "Espeak is not available");
}
#endif


std::vector<int32_t> vits_tokenizer::tokenize(const std::string& text) {
#ifdef VITS_ESPEAK
    if(!phonetic)
    {
#endif   
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
#ifdef VITS_ESPEAK
    }
    else
    {
	std::string copy(text);
	std::vector<char> phonetic_text = convert_to_phonetic(copy.data(), copy.data()+copy.size());
	std::vector<int32_t> tokens;
	tokens.reserve(phonetic_text.size()*(add_blank?2:1) + 1);
	if(add_blank)
	    tokens.push_back(0);
	std::basic_string_view v(phonetic_text.begin(), phonetic_text.end());
	while(!v.empty())
	{
	    auto it = vocab.begin();
	    while(it != vocab.end() && !v.starts_with(it->first))
		++it;
	    if(it == vocab.end())
	    {
		tokens.push_back(0);//did not find
		v.remove_prefix(1);
	    }
	    else
	    {
		tokens.push_back(it->second);
		v.remove_prefix(it->first.size());
	    }
	    if(add_blank)
		tokens.push_back(0);
	}
	return tokens;
    }
#endif  	
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