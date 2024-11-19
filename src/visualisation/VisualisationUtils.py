from LoggingService import LoggingService

def save_figure(plt_str, plt, model_name=None):
        directory = LoggingService().get_root_dir() + (f"/{model_name}" if model_name else "")
        plt.savefig(f"{directory}/{plt_str}.png")
        plt.close()