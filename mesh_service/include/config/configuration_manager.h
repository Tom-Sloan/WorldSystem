#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

namespace mesh_service {

/**
 * Singleton configuration manager for mesh service.
 * Loads configuration from environment variables and provides type-safe access.
 */
class ConfigurationManager {
private:
    static ConfigurationManager* instance_;
    static std::mutex mutex_;
    
    // Storage for different parameter types
    std::unordered_map<std::string, float> float_params_;
    std::unordered_map<std::string, int> int_params_;
    std::unordered_map<std::string, size_t> size_params_;
    std::unordered_map<std::string, std::string> string_params_;
    std::unordered_map<std::string, bool> bool_params_;
    
    // Private constructor
    ConfigurationManager() {
        loadFromEnvironment();
    }
    
    // Helper to convert string to numeric type
    template<typename T>
    T stringToNumber(const std::string& str, T default_val) const {
        std::istringstream iss(str);
        T value;
        if (iss >> value) {
            return value;
        }
        return default_val;
    }
    
public:
    // Delete copy constructor and assignment operator
    ConfigurationManager(const ConfigurationManager&) = delete;
    ConfigurationManager& operator=(const ConfigurationManager&) = delete;
    
    /**
     * Get singleton instance
     */
    static ConfigurationManager& getInstance() {
        // Double-checked locking pattern
        if (instance_ == nullptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (instance_ == nullptr) {
                instance_ = new ConfigurationManager();
            }
        }
        return *instance_;
    }
    
    /**
     * Load configuration from environment variables
     */
    void loadFromEnvironment();
    
    /**
     * Get float parameter
     */
    float getFloat(const std::string& key, float default_val) const {
        auto it = float_params_.find(key);
        if (it != float_params_.end()) {
            return it->second;
        }
        return default_val;
    }
    
    /**
     * Get integer parameter
     */
    int getInt(const std::string& key, int default_val) const {
        auto it = int_params_.find(key);
        if (it != int_params_.end()) {
            return it->second;
        }
        return default_val;
    }
    
    /**
     * Get size_t parameter
     */
    size_t getSize(const std::string& key, size_t default_val) const {
        auto it = size_params_.find(key);
        if (it != size_params_.end()) {
            return it->second;
        }
        return default_val;
    }
    
    /**
     * Get string parameter
     */
    std::string getString(const std::string& key, const std::string& default_val) const {
        auto it = string_params_.find(key);
        if (it != string_params_.end()) {
            return it->second;
        }
        return default_val;
    }
    
    /**
     * Get boolean parameter
     */
    bool getBool(const std::string& key, bool default_val) const {
        auto it = bool_params_.find(key);
        if (it != bool_params_.end()) {
            return it->second;
        }
        return default_val;
    }
    
    /**
     * Parse float3 from comma-separated string (e.g., "1.0,2.0,3.0")
     */
    bool parseFloat3(const std::string& str, float& x, float& y, float& z) const {
        std::istringstream iss(str);
        char comma1, comma2;
        if (iss >> x >> comma1 >> y >> comma2 >> z && comma1 == ',' && comma2 == ',') {
            return true;
        }
        return false;
    }
    
    /**
     * Log all loaded configuration values
     */
    void logConfiguration() const;
    
    /**
     * Validate configuration ranges
     */
    bool validateConfiguration() const;
    
    /**
     * Reset to defaults (mainly for testing)
     */
    void reset() {
        float_params_.clear();
        int_params_.clear();
        size_params_.clear();
        string_params_.clear();
        bool_params_.clear();
        loadFromEnvironment();
    }
};

// Convenience macros for easy access
#define CONFIG() mesh_service::ConfigurationManager::getInstance()
#define CONFIG_FLOAT(key, default_val) CONFIG().getFloat(key, default_val)
#define CONFIG_INT(key, default_val) CONFIG().getInt(key, default_val)
#define CONFIG_SIZE(key, default_val) CONFIG().getSize(key, default_val)
#define CONFIG_STRING(key, default_val) CONFIG().getString(key, default_val)
#define CONFIG_BOOL(key, default_val) CONFIG().getBool(key, default_val)

} // namespace mesh_service