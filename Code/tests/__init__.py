def test_radar_chart_with_positive_and_negative_values():
    # Setup radar chart with middle ring at 0
    radar_chart = RadarChart(middle_ring=0)
    radar_chart.add_data([5, -3, 2, -1, 4])
    
    # Render the chart
    rendered_chart = radar_chart.render()
    
    # Verify the chart contains both positive and negative values
    assert rendered_chart.contains_value(5)
    assert rendered_chart.contains_value(-3)
    assert rendered_chart.contains_value(2)
    assert rendered_chart.contains_value(-1)
    assert rendered_chart.contains_value(4)