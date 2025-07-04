def test_radar_chart_with_negative_values():
    # Assuming you have a function `render_radar_chart` that takes values and returns a chart object
    values = [-5, -3, 0, 2, 4]
    chart = render_radar_chart(values)
    
    assert chart.middle_ring == 0
    assert chart.values == values
    assert chart.has_negative_values() == True

def test_radar_chart_with_positive_values():
    values = [1, 2, 3, 4, 5]
    chart = render_radar_chart(values)
    
    assert chart.middle_ring == 0
    assert chart.values == values
    assert chart.has_negative_values() == False

def test_radar_chart_with_mixed_values():
    values = [-1, 0, 1, 2, -2]
    chart = render_radar_chart(values)
    
    assert chart.middle_ring == 0
    assert chart.values == values
    assert chart.has_negative_values() == True