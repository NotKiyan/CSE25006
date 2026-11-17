#ifndef CLIMATE_TRACE_H
#define CLIMATE_TRACE_H

#include "ns3/object.h"

#include <string>
#include <vector>

namespace ns3
{

struct ClimateSample
{
    double t;    //!< seconds since first sample
    double temp; //!< degC
    double hum;  //!< %RH
};

/**
 * Time series of temperature/humidity. Loaded from CSV (columns: timestamp,temp,hum;
 * timestamp = unix seconds) or generated synthetically. Linear interpolation, optional wrap-around.
 */
class ClimateTrace : public Object
{
  public:
    static TypeId GetTypeId();
    ClimateTrace();

    /// Rows with hum < humMin are dropped (sensor glitches, e.g. hum stuck at ~1%).
    bool LoadCsv(const std::string& path, double humMin = 5.0);

    /// Sinusoidal diurnal + seasonal temperature; humidity anti-correlated with temperature.
    void GenerateSynthetic(double days,
                           double stepSec,
                           double meanT,
                           double dailyAmp,
                           double seasonAmp,
                           double meanH,
                           double humPerC);

    void SetPeriodic(bool periodic) { m_periodic = periodic; }
    bool IsEmpty() const { return m_s.empty(); }
    double GetSpanSeconds() const { return m_s.empty() ? 0.0 : m_s.back().t; }

    /// Mean temperature/humidity over the whole trace (use as zero-offset reference).
    void GetMeans(double& temp, double& hum) const;

    /// t = trace seconds (already scaled/offset by the caller)
    void Get(double t, double& temp, double& hum) const;

  private:
    std::vector<ClimateSample> m_s;
    bool m_periodic{true};
};

} // namespace ns3

#endif
