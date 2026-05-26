#ifndef TENSORRT_CUSTOM_PLUGINS_TABLE_BUILDER_H
#define TENSORRT_CUSTOM_PLUGINS_TABLE_BUILDER_H

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <type_traits>
#include <tuple>

class TableBuilder {
public:
    TableBuilder& columns(const std::vector<std::string>& names) {
        m_headers = names;
        m_widths.resize(names.size());
        for (size_t i = 0; i < names.size(); ++i) {
            m_widths[i] = names[i].length();
        }
        return *this;
    }
    
    template<typename... Args>
    TableBuilder& row(Args&&... args) {
        std::vector<std::string> row_data;
        (row_data.push_back(format(std::forward<Args>(args))), ...);
        
        for (size_t i = 0; i < row_data.size(); ++i) {
            m_widths[i] = std::max(m_widths[i], (int)row_data[i].length());
        }
        m_rows.push_back(row_data);
        return *this;
    }
    
    void print() const {
        printLine();
        printHeader();
        printLine();
        for (const auto& row : m_rows) {
            printRow(row);
        }
        printLine();
    }
    
private:
    std::vector<std::string> m_headers;
    std::vector<std::vector<std::string>> m_rows;
    mutable std::vector<int> m_widths;
    
    template<typename T>
    std::string format(const T& value) const {
        std::ostringstream oss;
        if constexpr (std::is_floating_point_v<T>) {
            double v = static_cast<double>(value);
            if (std::abs(v) < 0.001 || std::abs(v) > 10000) {
                oss << std::scientific << std::setprecision(3) << v;
            } else {
                oss << std::fixed << std::setprecision(3) << v;
            }
        } else {
            oss << value;
        }
        return oss.str();
    }
    
    void printLine() const {
        std::cout << "+";
        for (int w : m_widths) {
            std::cout << std::string(w + 2, '-') << "+";
        }
        std::cout << "\n";
    }
    
    void printHeader() const {
        std::cout << "|";
        for (size_t i = 0; i < m_headers.size(); ++i) {
            std::cout << " " << std::left << std::setw(m_widths[i]) << m_headers[i] << " |";
        }
        std::cout << "\n";
    }
    
    void printRow(const std::vector<std::string>& row) const {
        std::cout << "|";
        for (size_t i = 0; i < row.size(); ++i) {
            if (i == 0) {
                std::cout << " " << std::left << std::setw(m_widths[i]) << row[i] << " |";
            } else {
                std::cout << " " << std::right << std::setw(m_widths[i]) << row[i] << " |";
            }
        }
        std::cout << "\n";
    }
};

#endif