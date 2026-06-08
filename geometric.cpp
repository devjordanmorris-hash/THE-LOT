// ===== PHASE 4.3 STEP 1 - PRISM + HUFFMAN (LITERAL ONLY) =====

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <queue>
#include <algorithm>

using namespace std;

/* ================= BASIC IO ================= */

void write_uint32(ofstream& out, uint32_t v) {
    out.write((char*)&v, sizeof(v));
}

uint32_t read_uint32(ifstream& in) {
    uint32_t v;
    in.read((char*)&v, sizeof(v));
    return v;
}

/* ================= BIT WRITER ================= */

struct BitWriter {
    vector<uint8_t> buffer;
    uint8_t current = 0;
    int bit_pos = 0;

    void write_bits(uint32_t bits, int count) {
        for (int i = 0; i < count; ++i) {
            current |= ((bits >> i) & 1) << bit_pos;
            bit_pos++;
            if (bit_pos == 8) {
                buffer.push_back(current);
                current = 0;
                bit_pos = 0;
            }
        }
    }

    void flush() {
        if (bit_pos > 0)
            buffer.push_back(current);
    }
};

/* ================= BIT READER ================= */

struct BitReader {
    const vector<uint8_t>& buffer;
    size_t index = 0;
    int bit_pos = 0;

    BitReader(const vector<uint8_t>& b) : buffer(b) {}

    uint32_t read_bit() {
        uint32_t bit = (buffer[index] >> bit_pos) & 1;
        bit_pos++;
        if (bit_pos == 8) {
            bit_pos = 0;
            index++;
        }
        return bit;
    }
};

/* ================= CSV PARSER ================= */

vector<string> parse_csv_line(const string& line) {
    vector<string> result;
    string field;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                i++;
            } else {
                in_quotes = !in_quotes;
            }
        }
        else if (c == ',' && !in_quotes) {
            result.push_back(field);
            field.clear();
        }
        else {
            field += c;
        }
    }

    result.push_back(field);
    return result;
}

/* ================= HUFFMAN ================= */

struct Node {
    uint32_t freq;
    int symbol;
    Node* left;
    Node* right;
};

struct Compare {
    bool operator()(Node* a, Node* b) {
        return a->freq > b->freq;
    }
};

void build_code_lengths(Node* node, int depth, vector<int>& lengths) {
    if (!node) return;
    if (node->symbol >= 0) {
        lengths[node->symbol] = depth;
        return;
    }
    build_code_lengths(node->left, depth + 1, lengths);
    build_code_lengths(node->right, depth + 1, lengths);
}

vector<int> generate_lengths(const vector<uint32_t>& freq) {

    priority_queue<Node*, vector<Node*>, Compare> pq;

    for (int i = 0; i < 256; ++i) {
        if (freq[i] > 0) {
            pq.push(new Node{freq[i], i, nullptr, nullptr});
        }
    }

    if (pq.empty())
        pq.push(new Node{1, 0, nullptr, nullptr});

    while (pq.size() > 1) {
        Node* a = pq.top(); pq.pop();
        Node* b = pq.top(); pq.pop();
        pq.push(new Node{a->freq + b->freq, -1, a, b});
    }

    Node* root = pq.top();
    vector<int> lengths(256, 0);
    build_code_lengths(root, 0, lengths);

    return lengths;
}

vector<uint32_t> build_canonical_codes(const vector<int>& lengths) {

    vector<pair<int,int>> symbols;
    for (int i = 0; i < 256; ++i)
        if (lengths[i] > 0)
            symbols.push_back({lengths[i], i});

    sort(symbols.begin(), symbols.end());

    vector<uint32_t> codes(256, 0);

    uint32_t code = 0;
    int prev_len = 0;

    for (auto& p : symbols) {
        int len = p.first;
        int sym = p.second;

        code <<= (len - prev_len);
        codes[sym] = code;
        code++;
        prev_len = len;
    }

    return codes;
}

/* ================= COMPRESS ================= */

void compress_csv(const string& input, const string& output) {

    ifstream in(input);
    ofstream out(output, ios::binary);

    vector<vector<string>> columns;
    string line;
    uint32_t row_count = 0;

    while (getline(in, line)) {

        auto parts = parse_csv_line(line);

        if (columns.empty())
            columns.resize(parts.size());

        for (size_t i = 0; i < parts.size(); ++i)
            columns[i].push_back(parts[i]);

        row_count++;
    }

    uint32_t col_count = columns.size();

    write_uint32(out, col_count);
    write_uint32(out, row_count);

    for (auto& col : columns) {

        string joined;
        for (size_t i = 0; i < col.size(); ++i) {
            joined += col[i];
            if (i != col.size() - 1)
                joined += '\n';
        }

        vector<uint32_t> freq(256, 0);
        for (unsigned char c : joined)
            freq[c]++;

        auto lengths = generate_lengths(freq);
        auto codes = build_canonical_codes(lengths);

        for (int i = 0; i < 256; ++i)
            out.put((uint8_t)lengths[i]);

        BitWriter bw;

        for (unsigned char c : joined)
            bw.write_bits(codes[c], lengths[c]);

        bw.flush();

        write_uint32(out, bw.buffer.size());
        out.write((char*)bw.buffer.data(), bw.buffer.size());
    }

    cout << "Compression complete\n";
}

/* ================= DECOMPRESS ================= */

void decompress_csv(const string& input, const string& output) {

    ifstream in(input, ios::binary);
    ofstream out(output);

    uint32_t col_count = read_uint32(in);
    uint32_t row_count = read_uint32(in);

    vector<vector<string>> columns(col_count);

    for (uint32_t c = 0; c < col_count; ++c) {

        vector<int> lengths(256);
        for (int i = 0; i < 256; ++i)
            lengths[i] = in.get();

        auto codes = build_canonical_codes(lengths);

        uint32_t bit_size = read_uint32(in);
        vector<uint8_t> buffer(bit_size);
        in.read((char*)buffer.data(), bit_size);

        BitReader br(buffer);

        string decoded;

        while (br.index < buffer.size()) {

            uint32_t code = 0;
            for (int len = 1; len < 32; ++len) {
                code |= br.read_bit() << (len - 1);

                for (int s = 0; s < 256; ++s) {
                    if (lengths[s] == len && codes[s] == code) {
                        decoded += (char)s;
                        goto next_symbol;
                    }
                }
            }
        next_symbol:;
        }

        string field;
        for (char ch : decoded) {
            if (ch == '\n') {
                columns[c].push_back(field);
                field.clear();
            } else {
                field += ch;
            }
        }
        columns[c].push_back(field);
    }

    for (uint32_t r = 0; r < row_count; ++r) {
        for (uint32_t c = 0; c < col_count; ++c) {

            string field = columns[c][r];

            bool need_quotes = false;
            for (char ch : field)
                if (ch == ',' || ch == '"')
                    need_quotes = true;

            if (need_quotes) {
                out << '"';
                for (char ch : field)
                    if (ch == '"') out << "\"\"";
                    else out << ch;
                out << '"';
            } else {
                out << field;
            }

            if (c != col_count - 1)
                out << ",";
        }
        if (r != row_count - 1)
            out << "\n";
    }

    cout << "Decompression complete\n";
}

/* ================= MAIN ================= */

int main(int argc, char* argv[]) {

    if (argc != 4) {
        cout << "Usage:\n";
        cout << "  geometric c input.csv output.prism\n";
        cout << "  geometric d input.prism output.csv\n";
        return 1;
    }

    if (string(argv[1]) == "c")
        compress_csv(argv[2], argv[3]);
    else
        decompress_csv(argv[2], argv[3]);

    return 0;
}
