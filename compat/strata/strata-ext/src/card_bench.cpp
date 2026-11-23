// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid D8 host half: statistics, pyramid ordering and the one-line report format.  See card_bench.hpp for the
// design gate.  Host-only: no GPU includes.
#include "strata/seabraid/card_bench.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

namespace seabraid {

SampleStats summarize(std::vector<double> v) {
    // Non-finite or non-positive times are measurement faults (a failed call, a clock that went backwards), not data.
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !(std::isfinite(x) && x > 0); }), v.end());
    SampleStats s;
    s.n = (int) v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    s.min = v.front();
    s.max = v.back();
    s.q1 = v[n / 4];
    s.q3 = v[std::min(n - 1, (3 * n) / 4)];
    s.median = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return s;
}

RegimeStat make_stat(int rows_per_call, int rows_per_expert, const std::vector<double>& seconds, int min_iters) {
    RegimeStat r;
    r.rows_per_call = rows_per_call;
    r.rows_per_expert = rows_per_expert;
    const SampleStats s = summarize(seconds);
    r.iters = s.n;
    if (rows_per_call <= 0 || s.n < std::max(1, min_iters)) return r;   // rows_per_s stays NaN: unmeasured
    r.rows_per_s = (double) rows_per_call / s.median;
    r.spread = (s.q3 - s.q1) / s.median;
    return r;
}

LeadStat make_lead(int streams, const std::vector<WindowSample>& windows, int min_iters) {
    std::vector<double> t;   // the timing boundary: expert_s is never read
    t.reserve(windows.size());
    for (const WindowSample& w : windows) t.push_back(w.nonexpert_s);
    const SampleStats s = summarize(t);
    LeadStat r;
    r.streams = streams;
    r.iters = s.n;
    if (streams <= 0 || s.n < std::max(1, min_iters)) return r;
    r.window_ms = s.median * 1e3;
    r.spread = (s.q3 - s.q1) / s.median;
    return r;
}

std::vector<int> order_lead(const std::vector<CardBench>& cards, LeadClass c, std::vector<int>* unmeasured) {
    std::vector<int> ok, bad;
    for (size_t i = 0; i < cards.size(); ++i) {
        if (cards[i].kind != TierKind::Main && cards[i].kind != TierKind::Helper) continue;
        (cards[i].lead_measured(c) ? ok : bad).push_back((int) i);
    }
    std::sort(ok.begin(), ok.end(), [&](int a, int b) {
        const CardBench &ca = cards[(size_t) a], &cb = cards[(size_t) b];
        const double ma = ca.lead(c).window_ms, mb = cb.lead(c).window_ms;
        if (ma != mb) return ma < mb;
        if (ca.free_vram_bytes != cb.free_vram_bytes) return ca.free_vram_bytes > cb.free_vram_bytes;
        return a < b;
    });
    if (unmeasured) *unmeasured = bad;
    ok.insert(ok.end(), bad.begin(), bad.end());
    return ok;
}

namespace {

// One class per pyramid level, best first.
constexpr int kLvlMain = 0, kLvlHelper = 1, kLvlRam = 2, kLvlNvme = 3, kLvlUnmeasured = 4;

int tier_level(const CardBench& c, Regime r) {
    switch (c.kind) {
        case TierKind::Main: return kLvlMain;
        case TierKind::Ram: return kLvlRam;
        case TierKind::Nvme: return kLvlNvme;
        default: return c.measured(r) ? kLvlHelper : kLvlUnmeasured;
    }
}

// Two measured helpers: -1 if a goes first, +1 if b does, 0 if they tie (rate desc, then free VRAM desc).
int cmp_helpers(const CardBench& ca, const CardBench& cb, Regime r) {
    const double ra = ca.stat(r).rows_per_s, rb = cb.stat(r).rows_per_s;
    if (ra != rb) return ra > rb ? -1 : 1;
    if (ca.free_vram_bytes != cb.free_vram_bytes) return ca.free_vram_bytes > cb.free_vram_bytes ? -1 : 1;
    return 0;
}

// The pyramid order: level first, helpers by cmp_helpers inside their level, the original index as the final key.
bool tier_before(const std::vector<CardBench>& cards, Regime r, int a, int b) {
    const CardBench &ca = cards[(size_t) a], &cb = cards[(size_t) b];
    const int xa = tier_level(ca, r), xb = tier_level(cb, r);
    if (xa != xb) return xa < xb;
    if (xa == kLvlHelper)
        if (const int c = cmp_helpers(ca, cb, r)) return c < 0;
    return a < b;
}

// Main/helper cards (in `order`) that have no measurement for `r`.
void collect_unmeasured(const std::vector<CardBench>& cards, const std::vector<int>& order, Regime r, std::vector<int>& out) {
    out.clear();
    for (int i : order) {
        const CardBench& c = cards[(size_t) i];
        if ((c.kind == TierKind::Main || c.kind == TierKind::Helper) && !c.measured(r)) out.push_back(i);
    }
}

}  // namespace

std::vector<int> order_tiers(const std::vector<CardBench>& cards, Regime r, std::vector<int>* unmeasured) {
    // The sort keys are (level, rate desc, free VRAM desc) with the original index as the final key, so the result is a
    // pure function of the input - no float noise, no unstable order.
    std::vector<int> idx(cards.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return tier_before(cards, r, a, b); });
    if (unmeasured) collect_unmeasured(cards, idx, r, *unmeasured);
    return idx;
}

namespace {
const char* kind_name(TierKind k) {
    return k == TierKind::Main ? "main" : k == TierKind::Ram ? "ram" : k == TierKind::Nvme ? "nvme" : "helper";
}
std::string esc(const std::string& s) {   // keep the line splittable on spaces and '='
    std::string o;
    for (unsigned char ch : s) {
        if (std::isalnum(ch) || ch == '.' || ch == '_' || ch == ':' || ch == '/' || ch == '+' || ch == '-') o += (char) ch;
        else { char b[4]; std::snprintf(b, sizeof b, "%%%02X", ch); o += b; }
    }
    return o;
}
std::string unesc(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && std::isxdigit((unsigned char) s[i + 1]) && std::isxdigit((unsigned char) s[i + 2])) {
            o += (char) std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
            i += 2;
        } else o += s[i];
    }
    return o;
}
std::string num(double v) {
    if (!std::isfinite(v)) return "nan";
    char b[32];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}
double dbl(const std::string& v) { return v == "nan" ? kNaN : std::strtod(v.c_str(), nullptr); }
void put_regime(std::string& o, const std::string& p, const RegimeStat& s) {
    o += " " + p + "_rows_per_s=" + num(s.rows_per_s) + " " + p + "_spread=" + num(s.spread) + " " + p +
         "_iters=" + std::to_string(s.iters) + " " + p + "_rows_per_call=" + std::to_string(s.rows_per_call) + " " + p +
         "_rows_per_expert=" + std::to_string(s.rows_per_expert);
}
void put_lead(std::string& o, const std::string& p, const LeadStat& s) {
    o += " " + p + "_window_ms=" + num(s.window_ms) + " " + p + "_spread=" + num(s.spread) + " " + p +
         "_iters=" + std::to_string(s.iters) + " " + p + "_streams=" + std::to_string(s.streams);
}
bool get_lead(LeadStat& s, const std::string& f, const std::string& v) {
    if (f == "window_ms") s.window_ms = dbl(v);
    else if (f == "spread") s.spread = dbl(v);
    else if (f == "iters") s.iters = std::atoi(v.c_str());
    else if (f == "streams") s.streams = std::atoi(v.c_str());
    else return false;
    return true;
}
bool get_regime(RegimeStat& s, const std::string& f, const std::string& v) {
    if (f == "rows_per_s") s.rows_per_s = dbl(v);
    else if (f == "spread") s.spread = dbl(v);
    else if (f == "iters") s.iters = std::atoi(v.c_str());
    else if (f == "rows_per_call") s.rows_per_call = std::atoi(v.c_str());
    else if (f == "rows_per_expert") s.rows_per_expert = std::atoi(v.c_str());
    else return false;
    return true;
}
TierKind kind_of(const std::string& v) {
    return v == "main" ? TierKind::Main : v == "ram" ? TierKind::Ram : v == "nvme" ? TierKind::Nvme : TierKind::Helper;
}
// The fields of a card line that belong to the card itself; false when `k` is not one of them.
bool set_scalar(CardBench& c, const std::string& k, const std::string& v) {
    if (k == "name") c.name = unesc(v);
    else if (k == "vendor") c.vendor = unesc(v);
    else if (k == "note") c.note = unesc(v);
    else if (k == "kind") c.kind = kind_of(v);
    else if (k == "free_vram_bytes") c.free_vram_bytes = std::strtoull(v.c_str(), nullptr, 10);
    else if (k == "layer") c.layer = std::atoi(v.c_str());
    else return false;
    return true;
}
// The decode_/prefill_/lead1_/lead4_ groups; an unknown key is ignored (a newer writer's field).
void set_prefixed(CardBench& c, const std::string& k, const std::string& v) {
    if (k.rfind("decode_", 0) == 0) get_regime(c.decode, k.substr(7), v);
    else if (k.rfind("prefill_", 0) == 0) get_regime(c.prefill, k.substr(8), v);
    else if (k.rfind("lead1_", 0) == 0) get_lead(c.lead_one, k.substr(6), v);
    else if (k.rfind("lead4_", 0) == 0) get_lead(c.lead_many, k.substr(6), v);
}
}  // namespace

std::string format_line(const CardBench& c) {
    const int ok = (int) c.measured(Regime::Decode) + (int) c.measured(Regime::Prefill);
    std::string o = "SEABRAID-CARD-BENCH v1 name=" + esc(c.name) + " vendor=" + esc(c.vendor) + " kind=" + kind_name(c.kind) +
                    " status=" + (ok == 2 ? "ok" : ok == 1 ? "partial" : "unmeasured") +
                    " free_vram_bytes=" + std::to_string(c.free_vram_bytes) + " layer=" + std::to_string(c.layer);
    put_regime(o, "decode", c.decode);
    put_regime(o, "prefill", c.prefill);
    put_lead(o, "lead1", c.lead_one);
    put_lead(o, "lead4", c.lead_many);
    if (!c.note.empty()) o += " note=" + esc(c.note);
    return o;
}

bool parse_line(const std::string& line, CardBench& out) {
    static const std::string kHead = "SEABRAID-CARD-BENCH v1 ";
    if (line.compare(0, kHead.size(), kHead) != 0) return false;
    CardBench c;
    size_t at = kHead.size();
    while (at < line.size()) {
        size_t end = line.find(' ', at);
        if (end == std::string::npos) end = line.size();
        const std::string tok = line.substr(at, end - at);
        at = end + 1;
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (!set_scalar(c, k, v)) set_prefixed(c, k, v);
    }
    out = c;
    return true;
}

}  // namespace seabraid
