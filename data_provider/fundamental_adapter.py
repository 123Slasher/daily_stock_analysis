# -*- coding: utf-8 -*-
"""
Fundamental adapter (Tushare-first, AkShare fallback).

This adapter tries Tushare Pro API first (if TUSHARE_TOKEN is configured),
then falls back to AkShare for Chinese financial data.
Never raises to caller; partial data is allowed.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_DIVIDEND_KEYWORD_MAP: Dict[str, List[str]] = {
    "per_share": [
        "每股派息",
        "每股现金红利",
        "每股分红",
        "每股派现",
        "派现(元/股)",
        "派息(元/股)",
        "税前派息(元/股)",
        "现金分红(税前)",
    ],
    "plan_text": [
        "分配方案",
        "分红方案",
        "实施方案",
        "派息方案",
        "方案",
        "预案",
        "方案说明",
    ],
    "ex_dividend_date": ["除权除息日", "除息日", "除权日", "除权除息", "除息日期"],
    "record_date": ["股权登记日", "登记日"],
    "announce_date": ["公告日期", "公告日", "实施公告日", "预案公告日"],
    "report_date": ["报告期", "报告日期", "截止日期", "统计截止日期"],
}


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort float conversion."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    try:
        return parsed.to_pydatetime()
    except Exception:
        return None


def _normalize_code(raw: Any) -> str:
    s = _safe_str(raw).upper()
    if "." in s:
        s = s.split(".", 1)[0]
    s = re.sub(r"^(SH|SZ|BJ)", "", s)
    return s


def _pick_by_keywords(row: pd.Series, keywords: List[str]) -> Optional[Any]:
    """
    Return first non-empty row value whose column name contains any keyword.
    """
    for col in row.index:
        col_s = str(col)
        if any(k in col_s for k in keywords):
            val = row.get(col)
            if val is not None and str(val).strip() not in ("", "-", "nan", "None"):
                return val
    return None


def _parse_dividend_plan_to_per_share(plan_text: str) -> Optional[float]:
    """Parse per-share cash dividend from Chinese plan text."""
    text = _safe_str(plan_text)
    if not text:
        return None

    for pattern in (
        r"(?:每)?\s*10\s*股?\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        r"10\s*派\s*([0-9]+(?:\.[0-9]+)?)\s*元",
    ):
        match = re.search(pattern, text)
        if match:
            parsed = _safe_float(match.group(1))
            if parsed is not None and parsed > 0:
                return parsed / 10.0

    match_per_share = re.search(r"每\s*股\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元", text)
    if match_per_share:
        parsed = _safe_float(match_per_share.group(1))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _extract_cash_dividend_per_share(row: pd.Series) -> Optional[float]:
    """Extract pre-tax cash dividend per share from a row."""
    plan_text = _safe_str(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["plan_text"]))
    if "税后" in plan_text and "税前" not in plan_text and "含税" not in plan_text:
        return None

    direct = _safe_float(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["per_share"]))
    if direct is not None and direct > 0:
        return direct
    return _parse_dividend_plan_to_per_share(plan_text)


def _filter_rows_by_code(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "symbol", "ts_code"))]
    if not code_cols:
        return df

    target = _normalize_code(stock_code)
    for col in code_cols:
        try:
            series = df[col].astype(str).map(_normalize_code)
            filtered = df[series == target]
            if not filtered.empty:
                return filtered
        except Exception:
            continue
    return pd.DataFrame()


def _normalize_report_date(value: Any) -> Optional[str]:
    parsed = _safe_datetime(value)
    return parsed.date().isoformat() if parsed else None


def _build_dividend_payload(
    dividend_df: pd.DataFrame,
    stock_code: str,
    max_events: int = 5,
) -> Dict[str, Any]:
    work_df = _filter_rows_by_code(dividend_df, stock_code)
    if work_df.empty:
        return {}

    now_date = datetime.now().date()
    ttm_start_date = now_date - timedelta(days=365)
    dedupe_keys = set()
    events: List[Dict[str, Any]] = []

    for _, row in work_df.iterrows():
        if not isinstance(row, pd.Series):
            continue
        ex_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["ex_dividend_date"]))
        record_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["record_date"]))
        announce_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["announce_date"]))
        event_dt = ex_dt or record_dt or announce_dt
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if event_date > now_date:
            continue

        per_share = _extract_cash_dividend_per_share(row)
        if per_share is None or per_share <= 0:
            continue

        dedupe_key = (event_date.isoformat(), round(per_share, 6))
        if dedupe_key in dedupe_keys:
            continue
        dedupe_keys.add(dedupe_key)

        events.append(
            {
                "event_date": event_date.isoformat(),
                "ex_dividend_date": ex_dt.date().isoformat() if ex_dt else None,
                "record_date": record_dt.date().isoformat() if record_dt else None,
                "announcement_date": announce_dt.date().isoformat() if announce_dt else None,
                "cash_dividend_per_share": round(per_share, 6),
                "is_pre_tax": True,
            }
        )

    if not events:
        return {}

    events.sort(key=lambda item: item.get("event_date") or "", reverse=True)
    ttm_events: List[Dict[str, Any]] = []
    for item in events:
        event_dt = _safe_datetime(item.get("event_date"))
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if ttm_start_date <= event_date <= now_date:
            ttm_events.append(item)

    return {
        "events": events[:max(1, max_events)],
        "ttm_event_count": len(ttm_events),
        "ttm_cash_dividend_per_share": (
            round(sum(float(item.get("cash_dividend_per_share") or 0.0) for item in ttm_events), 6)
            if ttm_events else None
        ),
        "coverage": "cash_dividend_pre_tax",
        "as_of": now_date.isoformat(),
    }


def _extract_latest_row(df: pd.DataFrame, stock_code: str) -> Optional[pd.Series]:
    """
    Select the most relevant row for the given stock.
    """
    if df is None or df.empty:
        return None

    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "ts_code", "symbol"))]
    target = _normalize_code(stock_code)
    if code_cols:
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                matched = df[series == target]
                if not matched.empty:
                    return matched.iloc[0]
            except Exception:
                continue
        return None

    return df.iloc[0]


def _ts_code(stock_code: str) -> str:
    """Convert 6-digit code to Tushare ts_code format (e.g. 600519 → 600519.SH)."""
    code = stock_code.strip().upper().replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    if code.startswith("6"):
        return f"{code}.SH"
    elif code.startswith(("0", "3")):
        return f"{code}.SZ"
    elif code.startswith(("4", "8")):
        return f"{code}.BJ"
    return code


def _get_tushare_token() -> Optional[str]:
    """Get Tushare token from environment or config, matching project convention."""
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    try:
        from src.config import get_config
        return (get_config().tushare_token or "").strip() or None
    except Exception:
        return None


class _TushareFundamentalClient:
    """Lightweight Tushare Pro client for fundamental data (no SDK required)."""

    def __init__(self, token: str, timeout: int = 30):
        self._token = token
        self._timeout = timeout
        self._api_url = "http://api.tushare.pro"

    def _query(self, api_name: str, fields: str = "", **kwargs) -> pd.DataFrame:
        req_params = {
            "api_name": api_name,
            "token": self._token,
            "params": kwargs,
            "fields": fields,
        }
        try:
            res = requests.post(self._api_url, json=req_params, timeout=self._timeout)
            if res.status_code != 200:
                raise Exception(f"Tushare HTTP {res.status_code}")
            result = _json.loads(res.text)
            if result.get("code") != 0:
                raise Exception(result.get("msg") or f"Tushare error code {result.get('code')}")
            data = result.get("data") or {}
            columns = data.get("fields") or []
            items = data.get("items") or []
            return pd.DataFrame(items, columns=columns)
        except Exception as e:
            logger.warning(f"Tushare {api_name} failed: {e}")
            raise

    def get_financial_indicators(self, stock_code: str) -> Optional[pd.DataFrame]:
        """Fetch fina_indicator (ROE, 毛利率, 营收同比, etc.) from Tushare."""
        ts = _ts_code(stock_code)
        fields = "ts_code,end_date,roe,roe_yoy,grossprofit_margin,or_yoy,profit_dedt,total_revenue,profit_to_op"
        try:
            df = self._query("fina_indicator", fields=fields, ts_code=ts, limit=4)
            if df.empty:
                return None
            # Sort by end_date descending
            if "end_date" in df.columns:
                df = df.sort_values("end_date", ascending=False)
            return df
        except Exception:
            return None

    def get_forecast(self, stock_code: str) -> Optional[str]:
        """Fetch earnings forecast from Tushare."""
        ts = _ts_code(stock_code)
        fields = "ts_code,ann_date,end_date,type,change_reason,p_change_min,p_change_max,net_profit_min"
        try:
            df = self._query("forecast", fields=fields, ts_code=ts, limit=1)
            if df.empty:
                return None
            row = df.iloc[0]
            forecast_type = _safe_str(row.get("type", ""))
            change_reason = _safe_str(row.get("change_reason", ""))
            p_min = _safe_float(row.get("p_change_min"))
            p_max = _safe_float(row.get("p_change_max"))
            parts = []
            if forecast_type:
                parts.append(f"类型: {forecast_type}")
            if p_min is not None and p_max is not None:
                parts.append(f"净利润变动: {p_min:.1f}%~{p_max:.1f}%")
            elif p_min is not None:
                parts.append(f"净利润变动: {p_min:.1f}%")
            if change_reason:
                parts.append(f"原因: {change_reason[:150]}")
            return "；".join(parts) if parts else None
        except Exception:
            return None

    def get_express(self, stock_code: str) -> Optional[str]:
        """Fetch quick earnings report from Tushare."""
        ts = _ts_code(stock_code)
        fields = "ts_code,ann_date,end_date,revenue,yoy,net_profit,yoy_np,eps"
        try:
            df = self._query("express", fields=fields, ts_code=ts, limit=1)
            if df.empty:
                return None
            row = df.iloc[0]
            revenue = _safe_float(row.get("revenue"))
            yoy = _safe_float(row.get("yoy"))
            net_profit = _safe_float(row.get("net_profit"))
            yoy_np = _safe_float(row.get("yoy_np"))
            parts = []
            if revenue is not None:
                parts.append(f"营收: {revenue/1e8:.2f}亿")
            if yoy is not None:
                parts.append(f"营收同比: {yoy:.1f}%")
            if net_profit is not None:
                parts.append(f"净利润: {net_profit/1e8:.2f}亿")
            if yoy_np is not None:
                parts.append(f"净利同比: {yoy_np:.1f}%")
            return "；".join(parts) if parts else None
        except Exception:
            return None

        def get_moneyflow(self, stock_code: str, trade_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
            """Fetch individual stock moneyflow from Tushare (主力资金流向)."""
            ts = _ts_code(stock_code)
            try:
                if not trade_date:
                    trade_date = datetime.now().strftime("%Y%m%d")
                fields = "ts_code,trade_date,buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount,net_mf_vol,net_mf_amount"
                df = self._query("moneyflow", fields=fields, ts_code=ts, trade_date=trade_date)
                if df is None or df.empty:
                    return None
                row = df.iloc[0]
                net_mf_amount = _safe_float(row.get("net_mf_amount"))
                net_mf_vol = _safe_float(row.get("net_mf_vol"))
                buy_lg = _safe_float(row.get("buy_lg_amount")) or 0
                sell_lg = _safe_float(row.get("sell_lg_amount")) or 0
                buy_elg = _safe_float(row.get("buy_elg_amount")) or 0
                sell_elg = _safe_float(row.get("sell_elg_amount")) or 0
                main_force_net = (buy_lg + buy_elg) - (sell_lg + sell_elg)
                return {
                    "trade_date": trade_date,
                    "net_mf_amount_wan": net_mf_amount,
                    "main_force_net_wan": main_force_net,
                    "buy_lg_amount_wan": buy_lg,
                    "sell_lg_amount_wan": sell_lg,
                    "buy_elg_amount_wan": buy_elg,
                    "sell_elg_amount_wan": sell_elg,
                }
            except Exception as e:
                logger.warning(f"Tushare moneyflow failed for {stock_code}: {e}")
                return None

class AkshareFundamentalAdapter:
    """Fundamental adapter: Tushare-first, AkShare fallback."""

    def __init__(self):
        self._ts_client: Optional[_TushareFundamentalClient] = None
        self._try_init_tushare()

    def _try_init_tushare(self) -> None:
        token = _get_tushare_token()
        if not token:
            logger.info("Tushare token not configured, will use AkShare only")
            return
        try:
            self._ts_client = _TushareFundamentalClient(token)
            logger.info("Tushare fundamental client initialized")
        except Exception as e:
            logger.warning(f"Tushare fundamental client init failed: {e}")
            self._ts_client = None

    def _get_fundamental_from_tushare(self, stock_code: str) -> Dict[str, Any]:
        """Fetch fundamental data from Tushare API."""
        result: Dict[str, Any] = {
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        if self._ts_client is None:
            return result

        # Financial indicators
        try:
            fin_df = self._ts_client.get_financial_indicators(stock_code)
            if fin_df is not None and not fin_df.empty:
                row = fin_df.iloc[0]
                roe = _safe_float(row.get("roe"))
                gross_margin = _safe_float(row.get("grossprofit_margin"))
                revenue_yoy = _safe_float(row.get("or_yoy"))
                profit_yoy = _safe_float(row.get("profit_dedt"))
                total_revenue = _safe_float(row.get("total_revenue"))
                net_profit_parent = _safe_float(row.get("profit_to_op"))
                end_date = _safe_str(row.get("end_date", ""))

                if any(v is not None for v in [roe, gross_margin, revenue_yoy, profit_yoy]):
                    result["growth"] = {
                        "revenue_yoy": revenue_yoy,
                        "net_profit_yoy": profit_yoy,
                        "roe": roe,
                        "gross_margin": gross_margin,
                    }
                    result["source_chain"].append("growth:tushare.fina_indicator")

                financial_report = {
                    "report_date": f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}" if len(end_date) >= 8 else end_date,
                    "revenue": total_revenue,
                    "net_profit_parent": net_profit_parent,
                    "roe": roe,
                }
                if any(v is not None for v in financial_report.values() if v != financial_report.get("report_date")):
                    result["earnings"]["financial_report"] = financial_report
        except Exception as e:
            result["errors"].append(f"tushare_fina:{e}")

        # Earnings forecast
        try:
            forecast = self._ts_client.get_forecast(stock_code)
            if forecast:
                result["earnings"]["forecast_summary"] = forecast[:200]
                result["source_chain"].append("earnings_forecast:tushare.forecast")
        except Exception as e:
            result["errors"].append(f"tushare_forecast:{e}")

        # Quick report
        try:
            express = self._ts_client.get_express(stock_code)
            if express:
                result["earnings"]["quick_report_summary"] = express[:200]
                result["source_chain"].append("earnings_quick:tushare.express")
        except Exception as e:
            result["errors"].append(f"tushare_express:{e}")

        return result

    def _call_df_candidates(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str]]:
        errors: List[str] = []
        try:
            import akshare as ak
        except Exception as exc:
            return None, None, [f"import_akshare:{type(exc).__name__}"]

        for func_name, kwargs in candidates:
            fn = getattr(ak, func_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                if isinstance(df, pd.Series):
                    df = df.to_frame().T
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df, func_name, errors
            except Exception as exc:
                errors.append(f"{func_name}:{type(exc).__name__}")
                continue
        return None, None, errors

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """
        Return normalized fundamental blocks.
        Tushare-first (API, works anywhere), AkShare fallback.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        # ---- Try Tushare first ----
        ts_result = self._get_fundamental_from_tushare(stock_code)
        if ts_result.get("growth") or ts_result.get("earnings"):
            result["growth"] = ts_result["growth"]
            result["earnings"] = ts_result["earnings"]
            result["source_chain"] = ts_result.get("source_chain", [])
            result["errors"] = ts_result.get("errors", [])
            has_content = bool(result["growth"] or result["earnings"])
            result["status"] = "partial" if has_content else "not_supported"
            if has_content:
                return result  # Tushare got data, skip AkShare for fundamentals

        # ---- Fallback: AkShare ----
        # Financial indicators
        fin_df, fin_source, fin_errors = self._call_df_candidates([
            ("stock_financial_abstract", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {}),
        ])
        result["errors"].extend(fin_errors)
        if fin_df is not None:
            row = _extract_latest_row(fin_df, stock_code)
            if row is not None:
                revenue_yoy = _safe_float(_pick_by_keywords(row, ["营业收入同比", "营收同比", "收入同比", "同比增长"]))
                profit_yoy = _safe_float(_pick_by_keywords(row, ["净利润同比", "净利同比", "归母净利润同比"]))
                roe = _safe_float(_pick_by_keywords(row, ["净资产收益率", "ROE", "净资产收益"]))
                gross_margin = _safe_float(_pick_by_keywords(row, ["毛利率"]))
                report_date = _normalize_report_date(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["report_date"]))
                revenue = _safe_float(_pick_by_keywords(row, ["营业总收入", "营业收入", "营收"]))
                net_profit_parent = _safe_float(_pick_by_keywords(row, ["归母净利润", "母公司股东净利润", "净利润"]))
                operating_cash_flow = _safe_float(
                    _pick_by_keywords(row, ["经营活动产生的现金流量净额", "经营现金流", "经营活动现金流"])
                )
                # Merge with Tushare data (Tushare may have partial, AkShare fills gaps)
                result["growth"] = {
                    "revenue_yoy": ts_result.get("growth", {}).get("revenue_yoy") or revenue_yoy,
                    "net_profit_yoy": ts_result.get("growth", {}).get("net_profit_yoy") or profit_yoy,
                    "roe": ts_result.get("growth", {}).get("roe") or roe,
                    "gross_margin": ts_result.get("growth", {}).get("gross_margin") or gross_margin,
                }
                financial_report_payload = {
                    "report_date": report_date,
                    "revenue": revenue,
                    "net_profit_parent": net_profit_parent,
                    "operating_cash_flow": operating_cash_flow,
                    "roe": roe,
                }
                # Merge: Tushare financial_report + AkShare extras
                ts_fin = ts_result.get("earnings", {}).get("financial_report") or {}
                merged = {**financial_report_payload, **ts_fin}
                merged = {k: v for k, v in merged.items() if v is not None}
                if merged:
                    result["earnings"]["financial_report"] = merged
                result["source_chain"].append(f"growth:{fin_source}")

        # Earnings forecast (AkShare)
        forecast_df, forecast_source, forecast_errors = self._call_df_candidates([
            ("stock_yjyg_em", {"symbol": stock_code}),
            ("stock_yjyg_em", {}),
            ("stock_yjbb_em", {"symbol": stock_code}),
            ("stock_yjbb_em", {}),
        ])
        result["errors"].extend(forecast_errors)
        if forecast_df is not None:
            row = _extract_latest_row(forecast_df, stock_code)
            if row is not None:
                # Only set if Tushare didn't already provide forecast
                if "forecast_summary" not in result["earnings"]:
                    result["earnings"]["forecast_summary"] = _safe_str(
                        _pick_by_keywords(row, ["预告", "业绩变动", "内容", "摘要", "公告"])
                    )[:200]
                result["source_chain"].append(f"earnings_forecast:{forecast_source}")

        # Earnings quick report (AkShare)
        quick_df, quick_source, quick_errors = self._call_df_candidates([
            ("stock_yjkb_em", {"symbol": stock_code}),
            ("stock_yjkb_em", {}),
        ])
        result["errors"].extend(quick_errors)
        if quick_df is not None:
            row = _extract_latest_row(quick_df, stock_code)
            if row is not None:
                if "quick_report_summary" not in result["earnings"]:
                    result["earnings"]["quick_report_summary"] = _safe_str(
                        _pick_by_keywords(row, ["快报", "摘要", "公告", "说明"])
                    )[:200]
                result["source_chain"].append(f"earnings_quick:{quick_source}")

        # Dividend details (cash dividend, pre-tax)
        dividend_df, dividend_source, dividend_errors = self._call_df_candidates([
            ("stock_fhps_detail_em", {"symbol": stock_code}),
            ("stock_history_dividend_detail", {"symbol": stock_code, "indicator": "分红", "date": ""}),
            ("stock_dividend_cninfo", {"symbol": stock_code}),
        ])
        result["errors"].extend(dividend_errors)
        if dividend_df is not None:
            dividend_payload = _build_dividend_payload(dividend_df, stock_code, max_events=5)
            if dividend_payload:
                result["earnings"]["dividend"] = dividend_payload
                result["source_chain"].append(f"dividend:{dividend_source}")

        # Institution / top shareholders
        inst_df, inst_source, inst_errors = self._call_df_candidates([
            ("stock_institute_hold", {}),
            ("stock_institute_recommend", {}),
        ])
        result["errors"].extend(inst_errors)
        if inst_df is not None:
            row = _extract_latest_row(inst_df, stock_code)
            if row is not None:
                inst_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "变动", "持股变化"]))
                result["institution"]["institution_holding_change"] = inst_change
                result["source_chain"].append(f"institution:{inst_source}")

        top10_df, top10_source, top10_errors = self._call_df_candidates([
            ("stock_gdfx_top_10_em", {"symbol": stock_code}),
            ("stock_gdfx_top_10_em", {}),
            ("stock_zh_a_gdhs_detail_em", {"symbol": stock_code}),
            ("stock_zh_a_gdhs_detail_em", {}),
        ])
        result["errors"].extend(top10_errors)
        if top10_df is not None:
            row = _extract_latest_row(top10_df, stock_code)
            if row is not None:
                holder_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "持股变化", "变动"]))
                result["institution"]["top10_holder_change"] = holder_change
                result["source_chain"].append(f"top10:{top10_source}")

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_capital_flow(self, stock_code: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Return stock + sector capital flow.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "stock_flow": {},
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": [],
            "errors": [],
        }

        stock_df, stock_source, stock_errors = self._call_df_candidates([
            ("stock_individual_fund_flow", {"stock": stock_code}),
            ("stock_individual_fund_flow", {"symbol": stock_code}),
            ("stock_individual_fund_flow", {}),
            ("stock_main_fund_flow", {"symbol": stock_code}),
            ("stock_main_fund_flow", {}),
        ])
        result["errors"].extend(stock_errors)
        if stock_df is not None:
            row = _extract_latest_row(stock_df, stock_code)
            if row is not None:
                net_inflow = _safe_float(_pick_by_keywords(row, ["主力净流入", "净流入", "净额"]))
                inflow_5d = _safe_float(_pick_by_keywords(row, ["5日", "五日"]))
                inflow_10d = _safe_float(_pick_by_keywords(row, ["10日", "十日"]))
                result["stock_flow"] = {
                    "main_net_inflow": net_inflow,
                    "inflow_5d": inflow_5d,
                    "inflow_10d": inflow_10d,
                }
                result["source_chain"].append(f"capital_stock:{stock_source}")

        sector_df, sector_source, sector_errors = self._call_df_candidates([
            ("stock_sector_fund_flow_rank", {}),
            ("stock_sector_fund_flow_summary", {}),
        ])
        result["errors"].extend(sector_errors)
        if sector_df is not None:
            name_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("板块", "行业", "名称", "name"))), None)
            flow_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("净流入", "主力", "flow", "净额"))), None)
            if name_col and flow_col:
                work_df = sector_df[[name_col, flow_col]].copy()
                work_df[flow_col] = pd.to_numeric(work_df[flow_col], errors="coerce")
                work_df = work_df.dropna(subset=[flow_col])
                top_df = work_df.nlargest(top_n, flow_col)
                bottom_df = work_df.nsmallest(top_n, flow_col)
                result["sector_rankings"] = {
                    "top": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in top_df.iterrows()],
                    "bottom": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in bottom_df.iterrows()],
                }
                result["source_chain"].append(f"capital_sector:{sector_source}")

        has_content = bool(result["stock_flow"] or result["sector_rankings"]["top"] or result["sector_rankings"]["bottom"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        """
        Return dragon-tiger signal in lookback window.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "is_on_list": False,
            "recent_count": 0,
            "latest_date": None,
            "source_chain": [],
            "errors": [],
        }

        df, source, errors = self._call_df_candidates([
            ("stock_lhb_stock_statistic_em", {}),
            ("stock_lhb_detail_em", {}),
            ("stock_lhb_jgmmtj_em", {}),
        ])
        result["errors"].extend(errors)
        if df is None:
            return result

        code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码"))]
        target = _normalize_code(stock_code)
        matched = pd.DataFrame()
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                cur = df[series == target]
                if not cur.empty:
                    matched = cur
                    break
            except Exception:
                continue
        if matched.empty:
            result["source_chain"].append(f"dragon_tiger:{source}")
            result["status"] = "ok" if code_cols else "partial"
            return result

        date_col = next((c for c in matched.columns if any(k in str(c) for k in ("日期", "上榜", "交易日", "time"))), None)
        parsed_dates: List[datetime] = []
        if date_col is not None:
            for val in matched[date_col].astype(str).tolist():
                try:
                    parsed_dates.append(pd.to_datetime(val).to_pydatetime())
                except Exception:
                    continue
        now = datetime.now()
        start = now - timedelta(days=max(1, lookback_days))
        recent_dates = [d for d in parsed_dates if start <= d <= now]

        result["is_on_list"] = bool(recent_dates)
        result["recent_count"] = len(recent_dates) if recent_dates else int(len(matched))
        result["latest_date"] = max(recent_dates).date().isoformat() if recent_dates else (
            max(parsed_dates).date().isoformat() if parsed_dates else None
        )
        result["status"] = "ok"
        result["source_chain"].append(f"dragon_tiger:{source}")
        return result
