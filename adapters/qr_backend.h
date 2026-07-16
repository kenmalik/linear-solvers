#pragma once

#include <cstdint>

enum class QrBackend : std::uint8_t { Householder,
                                      CholQR,
                                      CholQRDx };
