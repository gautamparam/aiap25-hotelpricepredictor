"""Shared fixtures.

The tests deliberately use small hand-written frames rather than the real
database. That keeps them fast, makes them runnable without `hotel.db`
present (so they work in a fresh clone), and — most importantly — means each
test states its own expected input explicitly instead of depending on
properties of a dataset the reader cannot see.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_bookings() -> pd.DataFrame:
    """A miniature raw bookings frame exercising every cleaning rule.

    Every column is TEXT, as in the source database. The rows are constructed
    to contain, between them: whitespace and casing variants, the documented
    typos, all four date formats, all six `is_canceled` spellings, a duplicate
    booking_id, and an implausible lead time.
    """
    return pd.DataFrame({
        "Unnamed: 0": ["0", "1", "2", "3", "4", "5"],
        "booking_id": ["SG1", "SG2", "SG3", "SG4", "SG5", "SG1"],
        "client_email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com", "a@x.com"],
        "hotel": ["City Hotel", "  City Hotel", "Airport Hotel", "AIRPORT HOTEL",
                  "City Hotel", "City Hotel"],
        "is_canceled": ["0", "1", "Yes", "No", "true", "0"],
        "lead_time": ["10", "200", "5", "-30", "4989", "10"],
        "arrival_date_year": ["2024", "2024", "2024", "2024", "2024", "2024"],
        "arrival_date_month": ["July", "Jul", "March", "Mar", "December", "July"],
        "arrival_date_week_number": ["27", "27", "10", "10", "50", "27"],
        "arrival_date_day_of_month": ["3", "4", "5", "6", "7", "3"],
        "stays_in_weekend_nights": ["1", "0", "2", "1", "0", "1"],
        "stays_in_week_nights": ["2", "3", "0", "1", "0", "2"],
        "adults": ["2", "1", "2", "2", "0", "2"],
        "children": ["0", "0", "1", "0", "0", "0"],
        "babies": ["0", "0", "0", "0", "0", "0"],
        "meal": ["BB", " BB ", "Undefined", "NULL", "SC", "BB"],
        "country": ["SGP", "Singapore", "CHN", "China", "", "SGP"],
        "market_segment": ["Online TA", "Onlin TA", "Offline TA/T0", "Groups",
                           "Direct", "Online TA"],
        "distribution_channel": ["TA/TO", " ta/to", "TA/TO", "Direct", "Direct", "TA/TO"],
        "is_repeated_guest": ["0", "0", "1", "0", "0", "0"],
        "previous_cancellations": ["0", "1", "0", "0", "0", "0"],
        "previous_bookings_not_canceled": ["0", "0", "3", "0", "0", "0"],
        "reserved_room_type": ["A", "A", "D", "A", "A", "A"],
        "assigned_room_type": ["A", "A", "D", "D", "A", "A"],
        "booking_changes": ["0", "1", "0", "2", "0", "0"],
        "deposit_type": ["No Deposit", "No Depost", "NON REFUND", "no-deposit",
                         "Refundable", "No Deposit"],
        "agent": ["9.0", "9.0", "", "240.0", "1.0", "9.0"],
        "company": ["", "", "Jurong Trading", "", "", ""],
        "days_in_waiting_list": ["0", "0", "0", "45", "0", "0"],
        "customer_type": ["Transient", "Transiant", "Transient-Party",
                          "Transient Party", "Contract", "Transient"],
        "adr": ["150.0", "200.5", "0", "300", "7108", "150.0"],
        "required_car_parking_spaces": ["0", "0", "1", "0", "0", "0"],
        "total_of_special_requests": ["1", "0", "2", "0", "0", "1"],
        "reservation_status": ["Check-Out", "Canceled", "Canceled", "Check-Out",
                               "No-Show", "Check-Out"],
        "reservation_status_date": ["2024-07-06 00:00:00", "03-13-24",
                                    "27/11/2023", "08 Mar 2024",
                                    "2024-12-07 00:00:00", "2024-07-06 00:00:00"],
        "promo_code": ["SAVE10", "", "LOYALTY", "", "FLASH", "SAVE10"],
    })
