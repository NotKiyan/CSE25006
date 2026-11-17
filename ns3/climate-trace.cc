#include "climate-trace.h"

#include "ns3/abort.h"
#include "ns3/log.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ClimateTrace");
NS_OBJECT_ENSURE_REGISTERED(ClimateTrace);

TypeId
ClimateTrace::GetTypeId()
{
    static TypeId tid = TypeId("ns3::ClimateTrace")
                            .SetParent<Object>()
                            .SetGroupName("Climate")
                            .AddConstructor<ClimateTrace>();
    return tid;
}

ClimateTrace::ClimateTrace() = default;

static std::string
Clean(std::string s)
{
    // strip UTF-8 BOM, whitespace, CR
    if (s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB && static_cast<unsigned char>(s[2]) == 0xBF)
    {
        s.erase(0, 3);
    }
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\n'))
    {
        s.pop_back();
    }
    while (!s.empty() && s.front() == ' ')
    {
        s.erase(0, 1);
    }
    return s;
}

static std::vector<std::string>
Split(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        out.push_back(Clean(tok));
    }
    if (!line.empty() && line.back() == ',')
    {
        out.emplace_back();
    }
    return out;
}

bool
ClimateTrace::LoadCsv(const std::string& path, double humMin)
{
    std::ifstream f(path);
    if (!f)
    {
        return false;
    }
    std::string line;
    if (!std::getline(f, line))
    {
        return false;
    }
    auto hdr = Split(line);
    int iT = -1;
    int iTemp = -1;
    int iHum = -1;
    for (size_t i = 0; i < hdr.size(); ++i)
    {
        if (hdr[i] == "timestamp")
        {
            iT = i;
        }
        else if (hdr[i] == "temp")
        {
            iTemp = i;
        }
        else if (hdr[i] == "hum")
        {
            iHum = i;
        }
    }
    if (iT < 0 || iTemp < 0 || iHum < 0)
    {
        return false;
    }
    std::vector<ClimateSample> rows;
    while (std::getline(f, line))
    {
        auto c = Split(line);
        if (c.size() <= static_cast<size_t>(std::max({iT, iTemp, iHum})))
        {
            continue;
        }
        try
        {
            ClimateSample s{std::stod(c[iT]), std::stod(c[iTemp]), std::stod(c[iHum])};
            if (s.hum < humMin)
            {
                continue;
            }
            rows.push_back(s);
        }
        catch (...)
        {
            continue;
        }
    }
    if (rows.empty())
    {
        return false;
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.t < b.t; });
    double t0 = rows.front().t;
    for (auto& r : rows)
    {
        r.t -= t0;
    }
    m_s = std::move(rows);
    NS_LOG_INFO("Loaded " << m_s.size() << " climate samples, span " << GetSpanSeconds() << " s");
    return true;
}

void
ClimateTrace::GenerateSynthetic(double days,
                                double stepSec,
                                double meanT,
                                double dailyAmp,
                                double seasonAmp,
                                double meanH,
                                double humPerC)
{
    m_s.clear();
    const double day = 86400.0;
    const double year = 365.0 * day;
    for (double t = 0; t <= days * day; t += stepSec)
    {
        double T = meanT + seasonAmp * std::sin(2 * M_PI * t / year) +
                   dailyAmp * std::sin(2 * M_PI * (t / day - 0.375)); // daily peak ~15:00
        double H = std::clamp(meanH - humPerC * (T - meanT), 5.0, 100.0);
        m_s.push_back({t, T, H});
    }
}

void
ClimateTrace::GetMeans(double& temp, double& hum) const
{
    NS_ABORT_MSG_IF(m_s.empty(), "ClimateTrace is empty");
    double st = 0;
    double sh = 0;
    for (const auto& x : m_s)
    {
        st += x.temp;
        sh += x.hum;
    }
    temp = st / m_s.size();
    hum = sh / m_s.size();
}

void
ClimateTrace::Get(double t, double& temp, double& hum) const
{
    NS_ABORT_MSG_IF(m_s.empty(), "ClimateTrace is empty: call LoadCsv or GenerateSynthetic");
    double span = m_s.back().t;
    if (m_periodic && span > 0)
    {
        t = std::fmod(t, span);
        if (t < 0)
        {
            t += span;
        }
    }
    if (t <= m_s.front().t)
    {
        temp = m_s.front().temp;
        hum = m_s.front().hum;
        return;
    }
    if (t >= m_s.back().t)
    {
        temp = m_s.back().temp;
        hum = m_s.back().hum;
        return;
    }
    auto it = std::upper_bound(m_s.begin(), m_s.end(), t, [](double v, const ClimateSample& s) {
        return v < s.t;
    });
    const auto& b = *it;
    const auto& a = *(it - 1);
    double w = (t - a.t) / (b.t - a.t);
    temp = a.temp + w * (b.temp - a.temp);
    hum = a.hum + w * (b.hum - a.hum);
}

} // namespace ns3
