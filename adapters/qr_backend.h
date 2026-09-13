#pragma once

#include <cstdint>

namespace cils {

enum class QrBackend : std::uint8_t { Householder,
                                      CholQR,
                                      CholQRDx };

}
