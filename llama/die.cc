#include "die.h"

#include <cstdio>
#include <cstdlib>
#include <print>

namespace l3llm {

void die_message(std::string_view message) {
    std::fflush(stdout); // keep partial progress output ahead of the error
    std::println(stderr, "l3llm: {}", message);
    std::exit(EXIT_FAILURE);
}

} // namespace l3llm
