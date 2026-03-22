# DONT WORRY IF THE IMPORTS ARE NOT FOUND, THEY EXECUTE INSIDE THE VENV
import os
from fpdf import FPDF

OUTPUT_DIR = "outputs/reports"
REPORT_TXT = os.path.join(OUTPUT_DIR, "data_summary.txt")
PDF_PATH   = os.path.join(OUTPUT_DIR, "weather_report.pdf")

# ──────────────────────────────────────────────────────────────
#  PLOT ORDER — reorder, remove, or add entries to control
#  what appears in the PDF and in what sequence.
# ──────────────────────────────────────────────────────────────
PLOT_ORDER = [
    (
        "Correlation Heatmap",
        "outputs/figures/bivariate/correlation_heatmap.png",
        '''This heatmap highlights relationships between weather variables. Strong correlations (e.g., temperature and feels-like temperature) appear in dark red, while negative correlations appear in dark blue. It helps identify which meteorological features are heavily interdependent, allowing us to create more insightful visualizations.'''
    ),
    (
        "Global Temperature Trend",
        "outputs/figures/temporal/global_warming_trend.png",
        "This line chart shows the average global temperature over the available dataset period. While long-term upward trajectories generally point toward the effects of global warming, the sharp declining trend seen here is contrary to expectations. This anomaly is explained by two main factors. First, global temperature trends are normally measured in decades rather than a span of a few years. Second, the dataset averages temperatures across all capital cities; because the vast majority of capital cities are located in the Northern Hemisphere, the data is heavily skewed. Consequently, instead of a true long-term climate shift, the chart primarily captures a natural seasonal transition from a Northern Hemisphere summer peak to a winter trough."
    ),
    (
        "BA - Rainfall by Season",
        "outputs/figures/temporal/ba_rain_by_season.png",
        "This series of charts breaks down rainfall in Buenos Aires by season. It highlights the city's wettest and driest periods, which are largely driven by regional atmospheric patterns and seasonal shifts in humidity. Interestingly, the data shows that spring, rather than summer, is the season with the highest accumulated rainfall."
    ),
    (
        "BA - Rain vs Wind Direction",
        "outputs/figures/temporal/ba_rain_vs_wind_direction.png",
        "This visualizes the relationship between wind direction and precipitation in Buenos Aires. While the data clearly shows that South and West winds are highly correlated with rainfall in Buenos Aires, this is driven by local atmospheric dynamics rather than the winds themselves acting as the primary source of moisture. Typically, the region experiences a buildup of warm, highly humid air brought by Northern winds. When a cold, dense front from the South or Southwest (a regional phenomenon known as the Pampero) moves in, it collides with this unstable air mass. The heavier cold air forces the warm moisture upward rapidly, causing sudden condensation and triggering heavy thunderstorms. Therefore, the high precipitation recorded during South and West winds captures the exact moment these cold fronts break the accumulated humidity, acting as a climatic trigger rather than a carrier of rain."
    ),
    (
        "BA vs Canberra Comparison",
        "outputs/figures/temporal/ba_vs_cape_town_comparison.png",
        "This dashboard compares the climate profiles of Buenos Aires and Canberra, two cities selected for their nearly identical latitudes (~35° South). While their shared latitude dictates perfectly synchronized seasonal cycles, the data reveals stark contrasts driven by their geographical contexts. Canberra exhibits a classic continental climate, characterized by wider temperature extremes—hotter summer peaks and colder winter valleys—and significantly lower, highly variable humidity. In contrast, Buenos Aires displays a narrower temperature distribution and consistently high humidity, with a median hovering near 80%. This comparison elegantly highlights how coastal proximity and oceanic thermal buffering (in Buenos Aires) versus an inland, elevated location (in Canberra) can drastically alter a region's climate despite receiving the exact same solar radiation."
    ),
    (
        "Temperature vs Pressure",
        "outputs/figures/bivariate/scatter_temperature_celsius__vs__pressure_mb.png",
        "This scatter plot illustrates the relationship between temperature and atmospheric pressure across a substantial dataset of over 130,000 observations. The negative Pearson correlation coefficient (r = -0.286) and the downward trend line confirm the physical principle that warmer air tends to be less dense, resulting in lower atmospheric pressure. However, the wide dispersion of data points and the relatively weak correlation value indicate that temperature is just one of many variables influencing pressure. Other geographic and meteorological factors strongly contribute to atmospheric pressure variations at any given temperature."
    ),
    (
        "Elevation vs Rainfall",
        "outputs/figures/geography/elevation_vs_rainfall.png",
        "While theoretical meteorology dictates that elevation influences precipitation, this scatter plot reveals that across global capital cities, elevation is not the sole driver of rainfall. Instead of a strong linear correlation, the data forms a dense central cluster. However, the visualization effectively highlights distinct outliers—such as Laos, Port Moresby, and Bandar Seri Begawan (predominantly in Asia and Oceania)—where extreme accumulated rainfall occurs at moderate estimated elevations. This suggests that while topography plays a localized role, macro-climatic factors like tropical monsoon seasons are the primary drivers of the world's most extreme precipitation levels."
    ),
    (
        "Air Quality PM2.5 Distribution",
        "outputs/figures/univariate/dist_air_quality_PM2_5.png",
        "This visualization illustrates the highly skewed global distribution of PM2.5 air pollution. As evidenced by the extreme positive skewness (8.98) and a mean (24.44) that is significantly pulled higher than the median (14.24), the vast majority of global records cluster within lower, healthier air quality ranges. However, the box plot reveals nearly 11,000 severe outliers, creating a massive 'long tail' in the dataset. These extreme values do not represent standard daily fluctuations, but rather acute environmental events—such as intense industrial emissions, seasonal crop burning, or geographic basins trapping smog. Ultimately, this demonstrates that while the 'typical' global air quality might appear moderate, the distribution is heavily punctuated by localized, severe pollution crises."
    ),
    (
        "Hemisphere Comparison",
        "outputs/figures/geography/hemisphere_comparison.png",
        "This dashboard highlights the structural climatic differences between the Northern and Southern Hemispheres. As the box plots demonstrate, the Northern Hemisphere exhibits extreme temperature volatility—evidenced by a much wider interquartile range and severe cold outliers dropping below -20°C. This is driven by its vast landmasses, which heat and cool rapidly. Conversely, the Southern Hemisphere shows a much tighter, more stable temperature distribution, reflecting the thermal buffering effect of its dominant oceans.\n""Furthermore, this visualization serves as a crucial data quality check: the Pressure chart reveals an impossible outlier in the Northern Hemisphere at approximately 3000 mb (standard sea-level pressure is ~1013 mb). This clearly indicates a sensor error or a corrupt data point in the raw dataset that requires cleaning."
    ),
    (
        "Air Quality Geographic Analysis",
        "outputs/figures/geography/air_quality_geographic_analysis.png",
        "This comprehensive dashboard provides a global overview of air quality, specifically focusing on PM2.5 particulate matter. The geographic analysis starkly identifies Asia as the most severely impacted continent, averaging a PM2.5 level of 43.8. This is heavily reflected in the 'Top 20 Most Polluted Capitals' list, which is overwhelmingly dominated by Asian and Middle Eastern cities. However, a significant anomaly emerges in the Americas: Santiago (Chile) ranks as the second most polluted capital globally, likely due to its unique topography (a valley surrounded by mountains) that traps smog. Furthermore, the correlation analysis confirms that PM2.5 concentration is heavily tied to other chemical pollutants (like Carbon Monoxide), while negatively correlated with humidity and cloud cover, indicating that wet, rainy conditions act as a natural cleanser for atmospheric particulate matter. "
    ),
    (
        "7-Day Temperature Forecast — Buenos Aires",
        "outputs/figures/forecast/forecast_7days.png",
        "This chart presents the predicted temperature for Buenos Aires over the next 7 days, generated using the Stacking Ensemble model (R2 = 0.928). The model combines Random Forest, Gradient Boosting, XGBoost, and LightGBM as base learners with a Ridge meta-learner. Predictions are based on the most recent weather observations for Buenos Aires from the dataset, with small day-to-day perturbations applied to simulate natural variability. Note: this is a model-based estimation, not a live meteorological forecast."
    ),
]


class WeatherPDF(FPDF):
    """Custom PDF with header/footer."""

    def header(self):
        if self.page_no() == 1:
            return  # cover page has its own layout
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Weather Forecasting Model - Report", align="R")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _add_cover(pdf: WeatherPDF):
    """Add a styled cover page."""
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, "Weather Forecasting Model", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Global Weather Repository - Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_draw_color(52, 152, 219)
    pdf.set_line_width(0.8)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Primary City: Buenos Aires", align="C", new_x="LMARGIN", new_y="NEXT")


def _add_text_report(pdf: WeatherPDF):
    """Embed the data summary text report across pages."""
    if not os.path.exists(REPORT_TXT):
        print(f"  ⚠️  Text report not found at {REPORT_TXT}")
        return

    with open(REPORT_TXT, "r", encoding="utf-8") as f:
        text = f.read()

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 12, "Data Summary", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Courier", "", 7)
    pdf.set_text_color(30, 30, 30)

    for line in text.split("\n"):
        # Replace characters that latin-1 can't encode
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.cell(0, 3.5, safe_line, new_x="LMARGIN", new_y="NEXT")


def _add_plot_page(pdf: WeatherPDF, title: str, img_path: str, conclusion: str):
    """Add a page with a titled plot image and conclusion."""
    if not os.path.exists(img_path):
        print(f"  ⚠️  Image not found, skipping: {img_path}")
        return

    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(44, 62, 80)
    safe_title = title.encode("latin-1", errors="replace").decode("latin-1")
    pdf.cell(0, 12, safe_title, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Image
    # Calculate width to fit the page with margins
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.image(img_path, x=pdf.l_margin*0.85, w=usable_w)
    
    # Move cursor below image
    pdf.ln(8)
    
    # Conclusion
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(44, 62, 80)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    
    # We replace any unsupported characters with a standard hyphen just in case
    safe_conclusion = conclusion.encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 5, safe_conclusion)


def run():
    """Generate the PDF report."""
    print("\n" + "=" * 60)
    print("  Phase: PDF REPORT GENERATION")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf = WeatherPDF(orientation="P", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # 1. Cover
    _add_cover(pdf)

    # 2. Text report
    _add_text_report(pdf)

    # 3. Plot pages in the configured order
    for title, img_path, conclusion in PLOT_ORDER:
        _add_plot_page(pdf, title, img_path, conclusion)
        print(f"  📄 Added: {title}")

    # Save
    pdf.output(PDF_PATH)
    print(f"\n  📕 PDF report saved to {PDF_PATH}")
