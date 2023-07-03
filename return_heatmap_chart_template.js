var {{chart_id}} = echarts.init(document.getElementById("{{chart_id}}"));

{{chart_id}}.setOption({
    tooltip: {
        position: 'top'
      },
      grid: {
        height: '75%',
		right: '10%',
        top: '2%'
      },
      xAxis: {
        name: 'Month',
        type: 'category',
        data: {{x_data}},
        splitArea: {
          show: true
        }
      },
      yAxis: {
        name: 'Year',
        nameLocation: 'middle',
        nameGap: '50',
        type: 'category',
        data: {{y_data}},
        splitArea: {
          show: true
        }
      },
      visualMap: {
        min: {{min_v}},
        max: {{max_v}},
        type: 'continuous',
        calculable: true,
        orient: 'vertical',
        top: '20%',
        right: '0.5%',
		color: ['orangered', 'lightgray', 'lightskyblue']
      },
      series: [
        {
          name: {{label}},
          type: 'heatmap',
          data: {{v_data}},
          label: {
            show: true
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
  });

window.addEventListener('resize',function(){
  {{chart_id}}.resize();
})