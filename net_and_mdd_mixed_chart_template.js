var {{chart_id}} = echarts.init(document.getElementById("{{chart_id}}"));

{{chart_id}}.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
        animation: false,
        label: {
          backgroundColor: '#505765'
        }
      }
    },
    grid: {
		top: '10%'
	},
    legend: {
      data: ['NAV', 'MDD'],
    },
    xAxis: [
      {
        type: 'category',
        boundaryGap: false,
        data: {{x_data}}
      }
    ],
    yAxis: [
      {
        name: 'NAV',
        nameLocation: 'middle',
        nameGap: '45',
        type: 'value',
        scale: true
      },
      {
        name: 'MDD',
        nameLocation: 'middle',
        nameGap: '50',
        type: 'value',
        scale: true,
        axisLabel: {
            formatter: '{value} %'
        },
        splitLine: {
            show: false
        }
      }
    ],
    series: [
      {
        yAxisIndex: 0,
        name: "NAV",
        type: 'line',
        data: {{y_data_1}}
      },
      {
        yAxisIndex: 1,
        name: "MDD",
        type: 'bar',
        data: {{y_data_2}},
        itemStyle: {
			color: '#ff5c5c',
			opacity: 0.1
		}
      }
    ]
  });

window.addEventListener('resize',function(){
  {{chart_id}}.resize();
})